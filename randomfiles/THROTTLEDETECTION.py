numAxes = self._joystick.get_numaxes()
jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        if (jsInputs[self._throttle_idx] != 0 and self._autopilot_enabled == True):
            self._autopilot_enabled = not self._autopilot_enabled
            world.player.set_autopilot(self._autopilot_enabled)
        elif (jsInputs[self._brake_idx] != 0 and self._autopilot_enabled == True):
            self._autopilot_enabled = not self._autopilot_enabled
            world.player.set_autopilot(self._autopilot_enabled)



numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        
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

        if (throttleCmd != 0 and self._autopilot_enabled == True):
            self._autopilot_enabled = False
            world.player.set_autopilot(self._autopilot_enabled)

        elif (brakeCmd != 0 and self._autopilot_enabled == True):
            self._autopilot_enabled = False
            world.player.set_autopilot(self._autopilot_enabled)