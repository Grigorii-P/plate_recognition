Покадровый прогон видео на распознавание номеров. Находим bboxes с помощью Mask-RCNN, затем распознаем текст в найденных прямоугольниках с помощью Tesseract OCR

## Скачивание необходимых компонентов

- устанавливаем ```tesseract 4.xx``` [(build from source)](https://github.com/tesseract-ocr/tesseract/wiki/Compiling)
- ставим зависимости ``` requirements.txt ```
- скачиваем модель для инференса [здесь](https://nomeroff.net.ua/models) и кладем в _Data/models_ (текущие используемые модели - ```imagenet_vgg16_np_region_2019_1_18.h5``` и ```mask_rcnn_numberplate_0700.h5```)
- тестовые видео скачиваем [здесь](https://drive.google.com/open?id=1gq14wGn3rotADueRI10ObOH1ZFEbsW5b) и кладем в *Data/video_results/preprocessed*

## Запуск
В ```run_detect.py``` выбираем видео, для которых хотим запустить распознавание:
```python
videos = {
# 'IMG_7460.MOV': IMG_7460,
# 'IMG_7461.MOV': IMG_7461,
# 'IMG_7462.MOV': IMG_7462,
# 'IMG_7463.MOV': IMG_7463,
# 'IMG_7464.MOV': IMG_7464,
# 'IMG_7465.MOV': IMG_7465,
# 'IMG_7466.MOV': IMG_7466,
'IMG_7467.MOV': IMG_7467,
'IMG_7501.MOV': IMG_7501,
'IMG_7502.MOV': IMG_7502,
'IMG_7503.MOV': IMG_7503,
'IMG_7504.MOV': IMG_7504,
'IMG_7505.MOV': IMG_7505,
'IMG_7506.MOV': IMG_7506,
}
```

Далее запускаем

```$(virtual_env) python run_detector.py```
