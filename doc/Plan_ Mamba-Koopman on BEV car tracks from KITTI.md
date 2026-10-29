# Mamba-Koopman on BEV car tracks from KITTI

The main part is the preparation of training data, this is directly related to 3D Vision techniques: multi-sensor calibration, camera–lidar fusion and bird’s-eye-view representation.
If data is good, Capstone should be considered successful even if training does not really work out. That said, a nice video showing predictions would be desirable.

## Training data preparation and visualisation

1. **Car’s positions relative to ego car, per video frame**
    - **a) From tracklets:** read `tracklet_labels.xml` and keep Car and Van positions, already in the Velodyne frame.
    - **b) Late Fusion:** detect and track cars in the camera image, then take each box’s position from the lidar points inside it (or from the ground plane?), in the Velodyne frame.

    > **Comments** on a) / b):
    > - **Alexander Haritonov** (2026-10-15): for simplicity, I'd take a)
    >   - **Jeremy Cohen** (2026-10-19): Ok. We can start this way and complexify later anyway

2. **Ego car’s pose (and camera pose) per video frame**
    - **a) From data:** the OXTS GPS/IMU pose per frame, chained with the fixed calibration (camera ↔ Velodyne ↔ IMU).
    - **b) From odometry:** when there’s no GPS/IMU, estimate it from the camera or from the lidar.

    > **Comments** on a) / b):
    > - **Alexander Haritonov** (2026-10-15): here also. just do a).  If the drive has it of course
    >   - **Jeremy Cohen** (2026-10-19): Yes, works for me
    >   - **Jeremy Cohen** (2026-10-19): kitti has oxts anyway everywhere, so b never makes sense here

3. **Convert to a world BEV:** combine 1 and 2 and drop height, so every car gets a 2D pose (x, y, θ) in one fixed top-down coordinate system per drive.

    > **Comment** on "drop height":
    > - **Jeremy Cohen** (2026-10-19): maybe reuse it later on without dropping it entirely

4. **Group into per-car sequences:** one position sequence per car ID. Smooth it and compute velocity and heading.

    > **Comment** on "Group into per-car sequences":
    > - **Jeremy Cohen** (2026-10-19): if something is occluded, you may want to define what to do with it
    >
    > **Comment** on "Smooth it and compute velocity and heading":
    > - **Jeremy Cohen** (2026-10-19): If you smooth the whole track before cutting samples, the last past positions already contain a bit of the future right?
    >
    >   Are you smoothing with all frames, or only the current position?
    >
    >   If all, may want to do 5 before 4.

5. **Cut into training samples:**
    - 10 positions in, 15 out.
    - Re-center on the target car’s pose (x_t, y_t, θ_t), facing +x.
    - Split by drive and save.

6. **Visualize training data.** One plot per coordinate frame:
    - **world BEV map:** all tracks of one drive. Parked cars must be dots, not smears, and you can put a number on it (position spread < ~0.2 m).
    - **camera image:** project world positions back into the video. The dots should land on the cars.
    - **target car frame:** Sample = one car at one moment t, with its 10 past positions and 15 future positions.
        - Pick randomly some samples. Plot each. In that car’s frame, facing +x.
        - Overlay samples with similar past positions. Take 1 randomly, find k most similar samples.

## Mamba-Koopman training

As per Claude, this shouldn't take much resources. I rented a simple gpu Scalway instance (for the reconstruction part of the cours), but would probably work even on a CPU. I do not need any accuracy here.

> **Comments** on "I do not need any accuracy here":
> - **Jeremy Cohen** (2026-10-19): Well, if you have a scene where most cars are parked; and your model randomly predicts 0 m/s motion, it'll get 100% accuracy. You still need a baseline to compare.
>   - **Alexander Haritonov** (2026-10-19): :)
>   - **Alexander Haritonov** (2026-10-19): baseline are the real tracks.
>
>     If I want a real experiment, I need to throw at least 1 drive out of training data.  Once NN is trained I run inference on that one drive, and compare the trajectory predictions with the recorded tracks.
>
>     For a good training I need also validation data - to know when to stop in order not to overfit.
>
>     It would be not clean for prod, but for demo simplest is reuse that same drive taken out of training.

## Output

Predicted bands. The model predicts 15 steps (frames), for each frame:

- Mu = (Mu_x, Mu_y) = averaged / mean car’s position.
- Sigma = (Sigma_x, Sigma_y) = spread.

Can be nicely visualized. Shoot video: two views side by side: BEV and camera (bands projected via the reverse coordinate transformation chain: target-car frame → world → Velodyne → camera → pixels).

## Review comments

> **Claude** (2026-10-29)

**Verdict:** the plan is solid and well scoped. Making data quality the success criterion is right, and the visualisation checks are the strongest part. The main risks are data volume and the missing baseline/metrics. The model section is still empty.

### Main concerns

1. **Data volume.** Drives 1, 9 and 14 have 134 Car/Van tracklets, but only 81 are ≥ 25 frames. Stride-1 windows give ~1,500 samples, but they overlap heavily: effectively ~80 independent tracks, many probably parked. Options:
    - Add more raw drives. Training only needs `tracklet_labels.xml`, OXTS and calib; images only for the demo drive.
    - Use the KITTI Tracking benchmark (21 labelled sequences with OXTS): more data in one standard format.
    - Use the ego car's own OXTS trajectory as extra moving-car samples. It's free and in every drive.
    - Use a shorter window (e.g. 10 in / 10 out) to keep more tracks.
2. **Baseline.** Agree with Jeremy: the real tracks are the *ground truth*, not a baseline. A baseline is a simple predictor the model has to beat:
    - **Constant position** and **constant velocity**. Constant velocity is notoriously hard to beat at 1.5 s.
    - The k-most-similar overlay in step 6 is also a ready-made **kNN predictor** baseline.
    - Metrics: ADE/FDE for μ, NLL or coverage for σ (~68% of the truth inside the 1σ band).
    - Report parked and moving cars separately, otherwise "predict standing still" looks great.
3. **Smoothing leakage.** Agree with Jeremy: centred smoothing over the whole track puts the future into the inputs, and inference has no future.
    - Don't smooth inputs, or smooth them causally only. Smoothing the *targets* is harmless.
    - Tracklets are already annotator-interpolated and may be clean enough as is. The parked-car check will tell.
4. **Heading.** Use the tracklet box yaw (`rz`), not the velocity direction. Velocity heading is undefined for parked cars, and θ_t is needed for re-centering in step 5.

### Smaller points

- **Step 3, drop height:** agree with Jeremy, keep z. Tracklet `tz` is the box bottom, and z is needed to project back into the camera. For predicted points, reuse the last known z.
- **Step 4, occlusion:** raw tracklets have no gaps (`first_frame` + `count`). Annotators interpolate through occlusions and set per-frame `occlusion`/`truncation` flags. So the rule can simply be "require 25 consecutive frames". The truncation flag also says whether the car is in the image at all, which matters for the camera view.
- **Validation:** reusing the held-out drive for early stopping is fine for a demo if stated. Cleaner: validate on held-out *track IDs* from the training drives, split by track, not by window (overlapping windows leak). Holding out drive 14 (~385 windows) leaves ~1,100 for training; holding out drive 9 would remove about two thirds of the data.
- **Model section:** define "Mamba-Koopman" concretely. E.g. a Mamba encoder reads the 10 past steps into a latent z, a linear Koopman matrix K rolls z forward 15 times, a decoder outputs (μ, σ), trained with Gaussian NLL.
    - Mamba's long-sequence advantage doesn't matter at 10 steps.
    - Keep the model tiny given ~80 tracks.
- **CPU:** the official `mamba_ssm` package needs CUDA. On CPU use a pure-PyTorch implementation; speed doesn't matter at this size.
- **Output:** a diagonal σ in the target-car frame makes sense (along-track vs. lateral). When transforming bands to world or camera, rotate the covariance: rotated ellipses, not axis-aligned boxes. σ should grow with the horizon; if not, something is wrong.

The parked-car "dots, not smears, < 0.2 m" check and the reprojection check are excellent: they test the whole calibration chain end to end, which is the 3D-vision core of the capstone.


