# DigitalMediaAnomalyPipeline
Metrics for detecting anomalies in both digital images and videos. More instructions to come...

# Requirements
<ul>
  <li>Python version 3.12 or below</li>
</ul>

# ImageAnomalyPipeline.py Instructions
<ul>
<li>python ImageAnomalyPipeline.py --mode check</li>
<li>python ImageAnomalyPipeline.py --mode clear</li>
<li>python ImageAnomalyPipeline.py --mode listmodels</li>
<li>python ImageAnomalyPipeline.py --mode all --model ResNet50 --dataset ucirvine_chest_xray --trainingseed 1 --inferenceseed 1 --epochs 5</li>
</ul>

# VideoAnomalyPipeline.py Instructions
<ul>
<li>python VideoAnomalyPipeline.py --mode check</li>
<li>python VideoAnomalyPipeline.py --mode clear</li>
<li>python VideoAnomalyPipeline.py --mode listmodels</li>
<li>python VideoAnomalyPipeline.py --mode all --model "CNN-LSTM" --dataset crime-ucf --trainingseed 1 --inferenceseed 1 --epochs 5</li>
</ul>

# Helper Python and Jupyter Notebook Scripts to execute ImageAnomalyPipeline.py and VideoAnomalyPipeline.py
<ul>
  <li>run_video_detection.py - helper CLI script for VideoAnomalyPipeline.py</li>
  <li>run_image_detection.py - helper CLI script for ImageAnomalyPipeline.py</li>
  <li>VideoAnomalyPipeline.ipynb - sample Jupyter Notebook script for VideoAnomalyPipeline.py</li>
  <li>ImageAnomalyPipeline.ipynb - sample Jupyter Notebook script for ImageAnomalyPipeline.py</li>
</ul>
