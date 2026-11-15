# Mamba-Koopman on BEV car tracks from KITTI

The main part is the preparation of training data, this is directly related to 3D Vision techniques: multi-sensor calibration, camera–lidar fusion and bird’s-eye-view representation.
If data is good, Capstone should be considered successful even if training does not really work out. That said, a nice video showing predictions would be desirable.

## Training data preparation and visualisation

1. **Car’s positions relative to ego car, per video frame**
    - **a) From tracklets:** read `tracklet_labels.xml` and keep Car, Van and Truck's positions, already in the Velodyne frame.
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

    > 2026-11-12: **file contract 1 of 3:** `prepared/vehicle_poses_world.csv`.
    > - Steps 4–6 and training read only `drive, track_id, object_type, frame, time, x, y, z, yaw, truncation`.
    > - Other data sources add rows in the same columns.
    > - Stages exchange data only through the three contract files.

    > 2026-11-14: the ego car is training data too. Its poses (step 2) become rows of the same table, one track per drive:
    > - `object_type` = `Ego`, `truncation` = 2 (never in the image).
    > - Position moved from the Velodyne to the centre of the car's bottom, as for the labeled cars.

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
    >
    >  (2026-11-11): skip smoothing and do not compute velocity; the model gets the raw positions.
    >   Step 4 is then only: group by track, require 25 consecutive frames.

5. **Cut into training samples:**
    - 10 positions in, 15 out.
    - Re-center on the target car’s pose (x_t, y_t, θ_t), facing +x.
    - Split by drive and save.

    2026-11-11:
    > - Save all windows of each split, without thinning.
    > - Build the baselines and metrics right after this step, before the model: constant position, constant velocity; ADE/FDE; parked and moving reported separately.
    > - 2026-11-14: report labeled cars and the ego car separately.

    2026-11-12, **file contract 2 of 3:** samples file, one `.npz` per split, with named arrays:
    > - `past` (N, input steps, 3) and `future` (N, output steps, 3): x, y, yaw in the target-car frame.
    > - `anchor_pose` (N, 4): x, y, z, yaw in the world.
    > - `drive`, `track_id`, `object_type`, `frame`.
    > - `displacement`, `yaw_change`: raw numbers per sample.
    > - `seconds_per_step`: one number.

    2026-11-12, rules:
    > - Input and output lengths are two arguments of the cutting script. Everywhere else they come from the array shapes.
    > - One module for the frame transform: world → car frame and car frame → world.
    > - The split is two explicit lists of drives (validation, test) in one file.

6. **Visualize training data.** One plot per coordinate frame:
    - **world BEV map:** all tracks of one drive. Parked cars must be dots, not smears, and you can put a number on it (position spread < ~0.2 m).
    - **camera image:** project world positions back into the video. The dots should land on the cars.
    - **target car frame:** Sample = one car at one moment t, with its 10 past positions and 15 future positions.
        - Pick randomly some samples. Plot each. In that car’s frame, facing +x.
        - Overlay samples with similar past positions. Take 1 randomly, find k most similar samples.

    > **Comment** on "find k most similar samples":
    > 2026-11-11: exclude the query car's own track (or drive) from the neighbours, here and in the kNN baseline.

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

> 2026-11-14: hold out two drives: 0011 for test and the video, 0018 for validation.
>   Freeze the split before the first training run.

> 2026-11-13: when the training script loads the samples:
> - Downsample parked windows and ego windows in the training set.
> - Use a larger stride for the validation set.

> 2026-11-11: with Gaussian NLL, clamp σ to a minimum of ~0.1 m, else parked cars can blow up the loss.

> 2026-11-12: the model is three separate pieces: encoder, rollout, decoder. The rollout takes the number of steps as an argument.
> - The decoder for μ is linear (one matrix). σ has a small nonlinear head.
> - The rollout is only a matrix multiplication with K: no activation or normalisation between the steps. K starts as the identity matrix.

> checks after training:
> - Plot the eigenvalues of K in the complex plane with the unit circle. They should be near or inside the circle.
> - Roll out 45 steps instead of 15 on a few test samples and plot. The path should continue smoothly and σ should keep growing.

> optional ablation: the same encoder with a plain network head that outputs all 15 steps at once, compared with the Koopman rollout.

> no consistency loss (encoding of a later window against the rolled-forward latent). Each sample is re-centred on its own car pose, so the two latents are in different frames.

## Output

Predicted bands. The model predicts 15 steps (frames), for each frame:

- Mu = (Mu_x, Mu_y) = averaged / mean car’s position.
- Sigma = (Sigma_x, Sigma_y) = spread.

> **Comment** on "Sigma":
> - **Claude** (2026-11-11): a single Gaussian per step is unimodal — at a crossing ("straight or turn") it gives one wide band between the options. Acceptable at 1.5 s; name it as a known limitation.

> 2026-11-12: **file contract 3 of 3:** predictions file with `mu` and `sigma`, both (N, output steps, 2), in the target-car frame, in the order of the samples file.
> - Baselines and the model write the same format.
> - Evaluation and video do not import the model.

Can be nicely visualized. Shoot video: two views side by side: BEV and camera (bands projected via the reverse coordinate transformation chain: target-car frame → world → Velodyne → camera → pixels).

> 2026-11-11: shoot the video on the held-out drive; avoid drives 0061, 0064, 0086 (height offset) and 0093 (no ego poses).

## Review comments

> **Claude** (2026-10-29, data numbers updated 2026-10-30)

### Main concerns

1. **Baseline.** Agree with Jeremy: the real tracks are the *ground truth*, not a baseline. A baseline is a simple predictor the model has to beat:
    - **Constant position** and **constant velocity**. Constant velocity is notoriously hard to beat at 1.5 s.
    - The k-most-similar overlay in step 6 is also a ready-made **kNN predictor** baseline.
    - Metrics: ADE/FDE for μ, NLL or coverage for σ (~68% of the truth inside the 1σ band).
    - Report parked and moving cars separately, otherwise "predict standing still" looks great.
2. **Smoothing leakage.** Agree with Jeremy: centred smoothing over the whole track puts the future into the inputs, and inference has no future.
    - Don't smooth inputs, or smooth them causally only. Smoothing the *targets* is harmless.
    - Tracklets are already annotator-interpolated and may be clean enough as is. The parked-car check will tell.
3. **Heading.** Use the tracklet box yaw (`rz`), not the velocity direction. Velocity heading is undefined for parked cars, and θ_t is needed for re-centering in step 5.

### Smaller points

- **Data:** the 38 raw drives with tracklets (all `2011_09_26`) give 12,919 frames and 742 Car/Van tracks ≥ 25 frames, so ~23,000 heavily overlapping stride-1 windows. The share of parked cars isn't measured yet. If moving cars turn out scarce, add the ego car's OXTS trajectories or shorten the window (e.g. 10 in / 10 out).
    - Measured 2026-11-11: of 22,875 windows, 38% moving and 62% parked; 177 of 709 tracks move. Enough moving data — keep 10 in / 15 out.
    - 2026-11-14: with moving = faster than 0.6 m/s over the window, labeled cars give 7,054 moving windows (31%) and 792 turning ones (over 15°). The ego car adds 10,502 moving and 1,160 turning.
- **Step 3, drop height:** agree with Jeremy, keep z. Tracklet `tz` is the box bottom, and z is needed to project back into the camera. For predicted points, reuse the last known z.
- **Step 4, occlusion:** raw tracklets have no gaps (`first_frame` + `count`). Annotators interpolate through occlusions and set per-frame `occlusion`/`truncation` flags. So the rule can simply be "require 25 consecutive frames". The truncation flag also says whether the car is in the image at all, which matters for the camera view.
- **Validation:** reusing the held-out drive for early stopping is fine for a demo if stated. Cleaner: validate on held-out *track IDs* from the training drives, split by track, not by window (overlapping windows leak). Holding out a drive now costs little: drive 9 (~1,000 windows) is under 5% of the data and drive 14 (~385 windows) under 2%, so several drives can be held out.
- **Model section:** define "Mamba-Koopman" concretely. E.g. a Mamba encoder reads the 10 past steps into a latent z, a linear Koopman matrix K rolls z forward 15 times, a decoder outputs (μ, σ), trained with Gaussian NLL.
    - Mamba's long-sequence advantage doesn't matter at 10 steps.
    - Keep the model small given ~740 tracks.
- **CPU:** the official `mamba_ssm` package needs CUDA. On CPU use a pure-PyTorch implementation; speed doesn't matter at this size.
- **Output:** a diagonal σ in the target-car frame makes sense (along-track vs. lateral). When transforming bands to world or camera, rotate the covariance: rotated ellipses, not axis-aligned boxes. σ should grow with the horizon; if not, something is wrong.


