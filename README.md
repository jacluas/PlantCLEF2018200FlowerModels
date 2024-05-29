# EfficientNet Models on PlantCLEF2018 200 Flower Dataset
### Repository of CNN models based on the EfficientNet Family of Architectures 

We share the EfficientNet models as research support in a subset of PlantCLEF2018 dataset.

The EfficientNet models were trained using transfer learning from pre-trained models on ImageNet 2012.

Models can be found here: [https://github.com/jacluas/PlantCLEF2018200FlowerModels/releases/tag/Release]

#### We believe that these models may be a good starting point for _transfer learning_ in future research on flower identification.

### Models were trained on a subset of flowering plant species of PlantCLEF2018 dataset

We created a subset of 200 classes, focusing on the most frequent flowering plant species on the PlantCLEF2018 dataset. 

### How to consume the EfficientNet models (from PlantCLEF 2018 200 Flower dataset)

We used Keras version 2.6, with Tensorflow 2.6 as backend, Python version 3.9, along with CUDA Toolkit version 11.0. 

The EfficientNet models report the percentage of correct responses within the _Top-K_ highest-ranked predictions generated.

You can evaluate a sample image by performing the following:

```python
python predict.py MODEL_NAME MODEL_PATH IMAGE_TEST_FILE TOP-K
```
To have the script automatically load the .bin file, use the same model name and ensure that the .bin file (class list) is in the same directory as the model.

Example _Top-5_:
```python
python predict.py efficientNetB0 /efficientNetB0PC/efficientNetB0PC /images/test/16730-Flower/197417.jpg -TopK 5

Predictions:
16730-Flower,	0.20647123456001282
44075-Flower,	0.1315551996231079
15859-Flower,	0.11214502155780792
152966-Flower,	0.07080081105232239
185502-Flower,	0.021940113976597786


```

