"""Which drives go into validation and test. All other drives are training.

Freeze these lists before the first training run.
"""

# Intersection like the test drive: 13 moving cars, 2 turning, the ego car turns
VALIDATION_DRIVES = ["2011_09_26_drive_0018"]

# Test and demo video: the most turning cars, up to 5 at once
TEST_DRIVES = ["2011_09_26_drive_0011"]
