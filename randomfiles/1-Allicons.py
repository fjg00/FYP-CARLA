#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.
Use ARROWS or WASD keys for control.
    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h
    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light
    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle
    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)
    R            : toggle recording images to disk
    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import time
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
#JASON
VIEW_WIDTH = 1920//3
VIEW_HEIGHT = 1080//3
VIEW_FOV = 90
check=""


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]


    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        #blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        print("BLUEP")
        #print(blueprint)
        blueprint= self.world.get_blueprint_library().filter('mercedesccc')[0]
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[0])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[1])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
           # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point= carla.Transform(carla.Location(x=-129.991348, y=0.500894, z=0.129238), carla.Rotation(pitch=0.405556, yaw=1.350928, roll=-0.013306))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            print(self.player)
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        #JASON
       # self.camera_manager.setup_depth_camera()
       # self.camera_manager.setup_depth2_camera()
        #JASON
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

    def tick(self, clock):
        self.hud.tick(self, clock)
    def render2(self,display):

        self.camera_manager.depth_render(display)
    def render3(self,display):
        self.camera_manager.depth2_render(display)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
#Jason
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
        self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
        self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
        self._parser.get('G29 Racing Wheel', 'handbrake'))

    def parse_events(self, client, world, clock):
        c = world.player.get_control()
        fyp  = c.throttle
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
                '''
            if self._autopilot_enabled == True and fyp != 0.0 :
                self._autopilot_enabled = not self._autopilot_enabled
                world.player.set_autopilot(self._autopilot_enabled)

                world.hud.notification(
                'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
'''
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3 :
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)

                    world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
                elif event.button == self._brake_idx and self._autopilot_enabled == True:
                    while (1):
                        print("hassoun")

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            
            if isinstance(self._control, carla.VehicleControl):
                #self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
                
                self._parse_vehicle_wheel()
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = throttleCmd
        self._control.throttle = brakeCmd

        toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])
    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]
        #self._steer_cache = 0
        #self._control.steer = 0
        #self._control.hand_brake = 0

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        #JASON
        #self.depth_camera = None
        #self.depth_display = None
        ##self.depth_image = None
        #self.depth_capture = True
        #self.depth = None
        #self.depth2_camera = None
        #self.depth2_display = None
        ##self.depth2_image = None
        #self.depth2_capture = True
        #self.depth2 = None
        #JASON
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-0.15,y=-0.4, z=1.2), carla.Rotation(yaw = 0)),Attachment.Rigid),
            (carla.Transform(carla.Location(x=-0.15,y=-0.4, z=1.2), carla.Rotation(yaw = 70)),Attachment.Rigid)]
    
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def depth_camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        depth_camera_bp = self._parent.get_world().get_blueprint_library().find('sensor.camera.rgb')
        depth_camera_bp.set_attribute('image_size_x', str(1920))
        depth_camera_bp.set_attribute('image_size_y', str(1080))
        depth_camera_bp.set_attribute('fov', str(VIEW_FOV))
        return depth_camera_bp

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

#JASON
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def Find_index(L,List):
    print(L)

    for i in range(len(List)):

        if List[i].id==L:
            return i

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        global Changed
        Changed=False
        Start=time.time()
        COUNT = 0
        client = carla.Client(args.host, args.port)
        tm = client.get_trafficmanager()  

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF| pygame.RESIZABLE)

        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        #world.world.wait_for_tick()
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()
        light_start=time.time()
        BOOL=False

        blueprint = random.choice(world.world.get_blueprint_library().filter('micra'))
        
        spawn_point4= carla.Transform(carla.Location(x=-54.605576, y=127.912247), carla.Rotation(pitch=0.007684, yaw=177.239975, roll=0.037449))
        p4=world.world.try_spawn_actor(blueprint, spawn_point4)
        #world.world.wait_for_tick()
        #spawn_point5= carla.Transform(carla.Location(x=-104.362762, y=136.437531), carla.Rotation(pitch=0.007438, yaw=-3.867036, roll=0.037474))
        #p5=world.world.try_spawn_actor(blueprint, spawn_point5)
        spawn_point6= carla.Transform(carla.Location(x=-104.093224, y=139.741302, z=0.000059), carla.Rotation(pitch=0.011755, yaw=-0.427490, roll=0.038036))
        p6=world.world.try_spawn_actor(blueprint, spawn_point6)
        #world.world.wait_for_tick()
        spawn_point7= carla.Transform(carla.Location(x=2.09, y=149.5), carla.Rotation(pitch=0, yaw=-90, roll=0.000004))
        spawn_point8= carla.Transform(carla.Location(x=-9.8, y=115.7), carla.Rotation(pitch=0, yaw=90, roll=0))
        p7=world.world.try_spawn_actor(blueprint, spawn_point7)
        p8=world.world.try_spawn_actor(blueprint, spawn_point8)
        spawn_point9= carla.Transform(carla.Location(x=7.4, y=-179.3), carla.Rotation(pitch=0, yaw=-90, roll=0))
        p9=world.world.try_spawn_actor(blueprint, spawn_point9)

        light_counter=0
        Traffic_comparison=0
        controller._autopilot_enabled=True
        world.player.set_autopilot(True)
        Traffic_light_id_list=[]
        Traffic_Light_id=-10
        Traffic_light_1= world.world.get_actors().filter('traffic.traffic_light')[1]
        print(len(world.world.get_actors().filter('traffic.traffic_light')))
        Traffic_light_1.set_state(carla.TrafficLightState.Red)
        Traffic_light_2= world.world.get_actors().filter('traffic.traffic_light')[7]
        Traffic_light_2.set_state(carla.TrafficLightState.Red)
        List=world.world.get_actors().filter('traffic.traffic_light')
        Traffic_light_1_id=   Traffic_light_1.id
        Traffic_light_2_id=   Traffic_light_2.id 
        boll=False
        
        while True:
            hud.notification(" FYP FYP FYP FYP")
            boll=-1
            Traffic_light_1.set_state(carla.TrafficLightState.Red)
            Traffic_light_2.set_state(carla.TrafficLightState.Red)
            print(world.player.get_transform().location.y)
            print(world.player.get_transform())        
            Traffic_Light=world.player.get_traffic_light()
            if Traffic_Light!=None:
                Traffic_Light_id=Traffic_Light.id 
            if Traffic_Light_id== Traffic_light_1_id or Traffic_Light_id==Traffic_light_2_id:
                tm.ignore_lights_percentage(world.player,100)
                light_start=time.time()
            else:
                tm.ignore_lights_percentage(world.player,0)
            print(Find_index(Traffic_Light_id,List))
            print(Traffic_Light)


            if controller._autopilot_enabled:
                world.player.set_autopilot(True)
                Auto = 1

                if (world.player.get_transform().location.x>-110 and world.player.get_transform().location.x<-95 and world.player.get_transform().location.y>-2 and world.player.get_transform().location.y<2): 
                    Timefirstintersection = time.time()                  
                    if List[36].state== carla.TrafficLightState.Red:                                                                         
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0                        
                    elif List[36].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1
                    
                elif (world.player.get_transform().location.x>163 and world.player.get_transform().location.x<177 and world.player.get_transform().location.y>-209 and world.player.get_transform().location.y<-205):
                    Timefourthintersection = time.time()                  
                    if List[20].state== carla.TrafficLightState.Red:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0
                    elif List[20].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1

                elif (world.player.get_transform().location.y>50 and world.player.get_transform().location.y<120 and world.player.get_transform().location.x>238 and world.player.get_transform().location.x<246):
                    Timethirdintersection = time.time()                  
                    if List[9].state== carla.TrafficLightState.Red:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0
                    elif List[9].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1                       

                elif (world.player.get_transform().location.x>85 and world.player.get_transform().location.x<112 and world.player.get_transform().location.y>-207 and world.player.get_transform().location.y<-203):
                    Timefifthintersection = time.time()                  
                    if List[24].state== carla.TrafficLightState.Red:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0
                    elif List[24].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1  
                
                elif (world.player.get_transform().location.x>1 and world.player.get_transform().location.x<33 and world.player.get_transform().location.y>-208 and world.player.get_transform().location.y<-206):
                    Timesixthintersection = time.time()                  
                    if boll==-1:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1 

                elif (world.player.get_transform().location.x>-150 and world.player.get_transform().location.x<-146 and world.player.get_transform().location.y>-25 and world.player.get_transform().location.y<-5):
                    Timeseventhintersection = time.time()                  
                    if List[32].state== carla.TrafficLightState.Red:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0
                    elif List[32].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1

                elif (world.player.get_transform().location.x>-115 and world.player.get_transform().location.x<-95 and world.player.get_transform().location.y>133 and world.player.get_transform().location.y<137):
                    Timeeighthintersection = time.time()                  
                    if List[3].state== carla.TrafficLightState.Red:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0
                    elif List[3].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1               

                elif (world.player.get_transform().location.x>203 and world.player.get_transform().location.x<225 and world.player.get_transform().location.y>60 and world.player.get_transform().location.y<64):
                    Timetenthintersection = time.time()                  
                    if List[10].state== carla.TrafficLightState.Red:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0                    
                    elif List[10].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1                   

                elif (world.player.get_transform().location.x>4 and world.player.get_transform().location.x<8 and world.player.get_transform().location.y>140 and world.player.get_transform().location.y<160):
                    Timeeleventhintersection = time.time()                  
                    if List[5].state== carla.TrafficLightState.Red:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Red Light Detected")
                        boll=0
                    elif List[5].state== carla.TrafficLightState.Green:
                        #hud.notification("No Pedestrian Detected - No Nearby Vehicle in the Blind Spot - Green Light Detected")
                        boll=1

                else:
                    boll=-1
######################################################################################################################################################################################################################################
                if (world.player.get_transform().location.x>-90 and world.player.get_transform().location.x<-84 and world.player.get_transform().location.y>100 and world.player.get_transform().location.y<120):
                    BOOOL=True                 
                    try:
                       p4.set_autopilot(True)
                       tm.ignore_lights_percentage(p4,100)
                    except:
                        print("")
                    try:
                       p6.set_autopilot(True)
                       tm.ignore_lights_percentage(p6,100)
                    except:
                       print("")       
#####################################################################################################################################################################################################################################

                if (world.player.get_transform().location.x>27 and world.player.get_transform().location.x<38 and world.player.get_transform().location.y>-210 and world.player.get_transform().location.y<-200):
                    BOOOL=True                 
                    try:
                       p9.set_autopilot(True)
                       tm.ignore_lights_percentage(p9,100)
                    except:
                        print("")
#####################################################################################################################################################################################################################################
                        #1st silent failure
                try:
                    if (world.player.get_transform().location.distance(p4.get_transform().location) < 8 or  world.player.get_transform().location.distance(p6.get_transform().location)< 8 ) and (time.time()-light_start)<10:
                        
                        start=time.time()
                        timefirstfailure = time.time()
                        world.player.set_autopilot(False)
                        a=carla.VehicleControl()
                        a.brake=1                        
                        a.throttle=0
                        world.player.apply_control(a)
                        controller._autopilot_enabled=False
                        Changed=True
                        Auto = 0
                    #elif (world.player.get_transform().location.x>-90 and world.player.get_transform().location.x<-84 and world.player.get_transform().location.y>70 and world.player.get_transform().location.y<180) and (time.time()-light_start)>15 and world.player.set_autopilot(False):

                        #Traffic_light_1.set_state(carla.TrafficLightState.Green)
        
                    else:
                        i="love"


                except:
                    i="love"
                else:
                    i="Love"
                try:
                    if (world.player.get_transform().location.x>-90 and world.player.get_transform().location.x<-84 and world.player.get_transform().location.y>70 and world.player.get_transform().location.y<180) and (time.time()-light_start)>15 and world.player.set_autopilot(False):
                        Traffic_light_1.set_state(carla.TrafficLightState.Green)
                except:
                    i="maya"
                else:
                    i="Maya"
                print("QWDQWD") 
#####################################################################################################################################################################################################################################
                if (world.player.get_transform().location.x>-33 and world.player.get_transform().location.x<-25 and world.player.get_transform().location.y>132 and world.player.get_transform().location.y<138):
                    BOOOL=True
                    try:
                        p7.set_autopilot(True)
                        tm.ignore_lights_percentage(p7,100)
                    except:
                        print("")
                    #try:
                        #p8.set_autopilot(True)
                        #tm.ignore_lights_percentage(p8,100)
                    #except:
                        #print("")
#####################################################################################################################################################################################################################################
                        #2nd silent failure
                try:
                    if (world.player.get_transform().location.distance(p7.get_transform().location) < 8 or  world.player.get_transform().location.distance(p8.get_transform().location)<8 ) and (time.time()-light_start)<10:
                        
                        start=time.time()
                        timesecondfailure = time.time()
                        world.player.set_autopilot(False)
                        a=carla.VehicleControl()
                        a.brake=1
                        a.throttle=0
                        world.player.apply_control(a)
                        controller._autopilot_enabled=False
                        Changed=True
                        Auto = 0
                    else:
                        i="love"
                except:
                    i="love"
                else:
                    i="Love"
                print("QWDQWD")
#####################################################################################################################################################################################################################################
            elif Changed and (time.time()-start)<5:
                print("FF")
                world.player.set_autopilot(False)
                a=carla.VehicleControl()
                a.brake=1
                a.throttle=0
                world.player.apply_control(a)
                controller._autopilot_enabled=False   
                 
            elif Changed:
                 world.player.set_autopilot(True)
                 controller._autopilot_enabled=True
                 Changed=False
                 Auto = 1
#####################################################################################################################################################################################################################################
            if BOOL:
                    try:
                       p4.set_autopilot(True)
                       tm.ignore_lights_percentage(p3,100)
                    except:
                        print("")
                    try:
                       p6.set_autopilot(True)
                       tm.ignore_lights_percentage(p6,100)
                    except:
                        print("")
                    try:
                       p7.set_autopilot(True)
                       tm.ignore_lights_percentage(p7,100)
                    except:
                        print("")                
                    try:
                       p8.set_autopilot(True)
                       tm.ignore_lights_percentage(p8,100)
                    except:
                        print("")
                      
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
##################################################################################################################################################################################################################################### ICONS CODE(FYP)
            if boll==-1:

                icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/WHITEicons.png')
        
            elif boll==0:    
            
                icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/REDicons.png')
              
            elif boll==1:
            
                icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/GREENicons.png')

            if controller._autopilot_enabled == False:
                Auto = 0 
    

            rect = icon.get_rect()
            rect.center = 1550, 870
            icon.convert()  
            display.blit(icon, rect)
            Gray = (128, 128, 128)
            pygame.draw.rect(display, Gray, rect, 2)  
            moving = False

            if Auto == 1:
                timeAutoOn = time.time()
                icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/ON2.png')
                display.blit(icon, (1550, 873))    
                pygame.display.set_icon(icon) 

            elif Auto == 0:
                timeAutoOff = time.time()
                if (world.player.get_transform().location.x>-108 and world.player.get_transform().location.x<-99 and world.player.get_transform().location.y>-1 and world.player.get_transform().location.y<2):
                    icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/OFF22.png')
                    display.blit(icon, (1550, 873))    
                    pygame.display.set_icon(icon) 
                elif (world.player.get_transform().location.x>-94 and world.player.get_transform().location.x<-83 and world.player.get_transform().location.y>-187 and world.player.get_transform().location.y<-142):
                    icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/OFF22.png')
                    display.blit(icon, (1550, 873))    
                    pygame.display.set_icon(icon) 
                elif (world.player.get_transform().location.x>169 and world.player.get_transform().location.x<171 and world.player.get_transform().location.y>77 and world.player.get_transform().location.y<93):
                    icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/OFF22.png')
                    display.blit(icon, (1550, 873))    
                    pygame.display.set_icon(icon) 
                elif (world.player.get_transform().location.x>210 and world.player.get_transform().location.x<227 and world.player.get_transform().location.y>61 and world.player.get_transform().location.y<66):
                    icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/OFF22.png')
                    display.blit(icon, (1550, 873))    
                    pygame.display.set_icon(icon) 
                elif (world.player.get_transform().location.x>11 and world.player.get_transform().location.x<36 and world.player.get_transform().location.y>192 and world.player.get_transform().location.y<194):
                    icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/OFF22.png')
                    display.blit(icon, (1550, 873))    
                    pygame.display.set_icon(icon) 
                elif (world.player.get_transform().location.x>5.9 and world.player.get_transform().location.x<6.6 and world.player.get_transform().location.y>138 and world.player.get_transform().location.y<149):
                    icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/OFF22.png')
                    display.blit(icon, (1550, 873))    
                    pygame.display.set_icon(icon) 
                else:
                    icon = pygame.image.load('/home/justin/Downloads/imagesCARLA/OFF21.png')
                    display.blit(icon, (1550, 873))    
                    pygame.display.set_icon(icon) 

                #display.blit(icon, (1550, 873))    
                #pygame.display.set_icon(icon) 

            pygame.display.flip()               

            end=time.time()
            elapsed=end-Start

            if (COUNT==0 or elapsed>1):
                
                file1 = open('/home/justin/Desktop/carla/ExperimentData/D1.1.txt', 'a')
                V=world.player.get_velocity()
                A=world.player.get_acceleration()
                C=world.player.get_control()
                steer=C.steer
                throttle=C.throttle
                brake=C.brake
                #reverse=C.reverse
                #hand_brake=C.hand_brake
                #manual=C.manual_gear_shift

                speed=3.6 * math.sqrt(V.x**2 + V.y**2 + V.z**2)
                acc=math.sqrt(A.x**2 + A.y**2 + A.z**2)
                file1.write("Time: "+str(Start)+"\n")
                file1.write("Acceleration: "+str(acc)+"\n")
                file1.write("Speed: " + str(speed)+"\n")
                file1.write("throttle: "+str(throttle)+"\n")
                file1.write("Steer: "+str(steer)+"\n")
                file1.write("brake: "+str(brake)+"\n")
                #file1.write("reverse: "+str(reverse)+"\n")
                #file1.write("hand_brake: "+str(hand_brake)+"\n")
                #file1.write("manual: "+str(manual)+"\n")

                transform = world.player.get_transform()
                
                heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
                heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
                heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
                heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
                Rotation = transform.rotation.yaw
                Heading = heading
                Location = transform.location.x, transform.location.y
                Height = transform.location.z

                file1.write("Heading: "+str(Heading)+"\n")
                file1.write("Location: "+str(Location)+"\n")
                file1.write("Height: " + str(Height)+"\n")
                file1.write("Rotation: " + str(Rotation)+"\n")
          
                if (world.player.get_transform().location.distance(p4.get_transform().location) < 8 or  world.player.get_transform().location.distance(p6.get_transform().location)< 8 ) and (time.time()-light_start)<10:
                    file1.write("Time of 1st silent failure (2nd intersection): "+str(timefirstfailure)+"\n")
                if (world.player.get_transform().location.distance(p7.get_transform().location) < 8 or  world.player.get_transform().location.distance(p8.get_transform().location)<8 ) and (time.time()-light_start)<10:
                    file1.write("Time of 2nd silent failure (9th intersection): "+str(timesecondfailure)+"\n")
                if Auto == 1:
                    file1.write("timeAutoOn: "+str(timeAutoOn)+"\n")
                if Auto == 0:
                    file1.write("timeAutoOff: "+str(timeAutoOff)+"\n")
                if (world.player.get_transform().location.x>-110 and world.player.get_transform().location.x<-95 and world.player.get_transform().location.y>-2 and world.player.get_transform().location.y<2): 
                    try:
                        file1.write("Time of the 1st intersection: "+str(Timefirstintersection)+"\n")
                    except:
                        print("")
                    if List[36].state== carla.TrafficLightState.Red:                                                                         
                        file1.write("Light State: Red"+"\n")                       
                    elif List[36].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n")      
                elif (world.player.get_transform().location.x>163 and world.player.get_transform().location.x<177 and world.player.get_transform().location.y>-209 and world.player.get_transform().location.y<-205):
                    file1.write("Time of the 4th intersection: "+str(Timefourthintersection)+"\n")   
                    if List[20].state== carla.TrafficLightState.Red:
                        file1.write("Light State: Red"+"\n")        
                    elif List[20].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n")
                elif (world.player.get_transform().location.y>50 and world.player.get_transform().location.y<120 and world.player.get_transform().location.x>238 and world.player.get_transform().location.x<246):
                    file1.write("Time of the 3rd intersection: "+str(Timethirdintersection)+"\n")
                    if List[9].state== carla.TrafficLightState.Red:
                        file1.write("Light State: Red"+"\n")        
                    elif List[9].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n") 
                elif (world.player.get_transform().location.x>85 and world.player.get_transform().location.x<112 and world.player.get_transform().location.y>-207 and world.player.get_transform().location.y<-203):
                    file1.write("Time of the 5th intersection: "+str(Timefifthintersection)+"\n")
                    if List[24].state== carla.TrafficLightState.Red:
                        file1.write("Light State: Red"+"\n")        
                    elif List[24].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n") 
                elif (world.player.get_transform().location.x>1 and world.player.get_transform().location.x<33 and world.player.get_transform().location.y>-208 and world.player.get_transform().location.y<-206):
                    file1.write("Time of the 6th intersection: "+str(Timesixthintersection)+"\n")
                    file1.write("Light State: Green"+"\n")
                elif (world.player.get_transform().location.x>-150 and world.player.get_transform().location.x<-146 and world.player.get_transform().location.y>-25 and world.player.get_transform().location.y<-5):
                    file1.write("Time of the 7th intersection: "+str(Timeseventhintersection)+"\n")
                    if List[32].state== carla.TrafficLightState.Red:
                        file1.write("Light State: Red"+"\n")        
                    elif List[32].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n")
                elif (world.player.get_transform().location.x>-115 and world.player.get_transform().location.x<-95 and world.player.get_transform().location.y>133 and world.player.get_transform().location.y<137):
                    file1.write("Time of the 8th intersection: "+str(Timeeighthintersection)+"\n")
                    if List[3].state== carla.TrafficLightState.Red:
                        file1.write("Light State: Red"+"\n")        
                    elif List[3].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n")   
                elif (world.player.get_transform().location.x>203 and world.player.get_transform().location.x<225 and world.player.get_transform().location.y>60 and world.player.get_transform().location.y<64):
                    file1.write("Time of the 10th intersection: "+str(Timetenthintersection)+"\n")
                    if List[10].state== carla.TrafficLightState.Red:
                        file1.write("Light State: Red"+"\n")                           
                    elif List[10].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n")   
                elif (world.player.get_transform().location.x>4 and world.player.get_transform().location.x<8 and world.player.get_transform().location.y>140 and world.player.get_transform().location.y<160):
                    file1.write("Time of the 11th intersection: "+str(Timeeleventhintersection)+"\n")
                    if List[5].state== carla.TrafficLightState.Red:
                        file1.write("Light State: Red"+"\n")        
                    elif List[5].state== carla.TrafficLightState.Green:
                        file1.write("Light State: Green"+"\n")
                else:
                    file1.write("Light State: No Light Detected"+"\n")

                Start=time.time()
                COUNT+=1
                
            print(world.player.get_transform())
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)


    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    #withoutnotifications
    argparser.add_argument(
              '--res',
              metavar='WIDTHxHEIGHT',
              default='1920x1080',
              help='window resolution (default: 1920x1080)')
    '''argparser.add_argument(
                    '--res',
                    metavar='WIDTHxHEIGHT',
                    default='1280x720',
                    help='window resolution (default: 1280x720)')'''
    #Hiba
    '''argparser.add_argument(
                    '--res',
                    metavar='WIDTHxHEIGHT',
                    default='1850x950',
                    help='window resolution (default: 1850x950)')'''

    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()


