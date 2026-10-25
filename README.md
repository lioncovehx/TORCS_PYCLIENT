# TORCS Python Client

A Python bot that drives a car in the [TORCS](https://sourceforge.net/projects/torcs/) racing simulator using a neural network. I recorded myself driving with the keyboard, then trained a model to copy what I did.

It talks to TORCS through the SCR server over UDP. Every game tick it gets the car's sensor readings and sends back steering, throttle, brake and gear.

## How it works

1. **Collect data.** `driver_manual.py` lets you drive with the arrow keys and logs every tick to a CSV: the 19 track-edge range sensors, 36 opponent sensors, speed, RPM, wheel spin, angle to the track and position on it, plus the controls you pressed. The training set is about 107k ticks (`merged_data.csv.gz`).
2. **Train.** `train.py` normalises the 66 input features and trains `DrivingModel` (in `model.py`) with an 80/20 train/val split. It keeps the checkpoint with the best validation loss.
3. **Drive.** `driver.py` loads `best_model.pth` and the saved feature means and stds, then drives the car on its own.

The model has a small sub-network for each sensor group (track, opponents, speed, wheels) and merges them in shared layers. It ends in four separate heads:

- steering: regression, MSE loss
- throttle and brake: sigmoid outputs, BCE loss
- gear: 8-way classification (reverse to 6th)

Steering and gear get higher loss weights, because those were the outputs that kept going wrong. There are also small penalty terms against jumpy gear changes and against braking and accelerating at the same time.

## Run it

```bash
pip install -r requirements.txt
```

Start TORCS with the SCR server patch and pick an `scr_server` car. Then:

```bash
python pyclient.py              # AI driver, connects to localhost:3001
python pyclient.py --port 3002  # second car slot
```

A trained model (`best_model.pth`, `means.npy`, `stds.npy`) is already in the repo, so you can skip training.

**Retrain:**

```bash
python train.py
```

**Record your own driving:** in `pyclient.py`, change `import driver` to `import driver_manual as driver`, run it, and drive with the arrow keys. Telemetry goes to the CSV named at the top of `driver_manual.py`.

## Files

| File | What it does |
|---|---|
| `pyclient.py` | UDP client, the entry point |
| `driver.py` | AI driver, runs the model |
| `driver_manual.py` | Keyboard driver for recording data |
| `model.py` | The network and its loss |
| `train.py` | Training loop |
| `carState.py`, `carControl.py`, `msgParser.py` | Parse sensor messages and build control messages (from the SCR client template) |
