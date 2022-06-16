# Traffic sign detection project

Installation:
- Pull the entire project (folders other than the `/Projekt` were used for lab collaboration and aren't a part of the project)
- Install dependencies: `pip install -r requirements.txt`
- Unzip `cnn_model_rgb_signs_long.zip` into models and update the folder name to match the zip name, i.e. `models/cnn_model_rgb_signs_long`

Most scripts can be configured to use different images by editing paths in the `.py` file. Runnable scripts (i.e. `python <script filename>`) are:
- `classifiers_signs_test.py` - Runs an automated test with samples from a folder configured inside. Shows incorrectly marked images.
- `contour.py` - Demo of the contour detection method
- `detection_sliding_window.py` - Demo of the sliding window detection method
- `main_detector.py` - The main detector described in the report
- `main_detector_vid.py` - Main detector running on a video example. Might cause dependency issues because of codecs

`train_model.py` was used as a way of sharing code for Google Colab notebooks and might not be up to date with the final results.
