`host_pwm_position.py` flies either controller, selected with `--controller`:

```
python host_pwm_position.py --controller pid        # default
python host_pwm_position.py --controller augmented
```

It has the following keyboard controls:
`Space` kills all motors, and terminates the script. **This only works if you are actively clicked into the terminal**
`1`, `2`, `3` and `4` all reduce the speed of one motor depending on the key pressed. The speed multiplier that the motors are set to can be changed by changing the `REDUCE_MULTIPLIER` value.
