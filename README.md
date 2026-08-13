# DigitalMediaAnomalyPipeline
Metrics for detecting anomalies in both digital images and videos. More instructions to come...

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
