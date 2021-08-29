from rest_framework import serializers

from rest_framework.serializers import (
      ModelSerializer,
)
from django.core.files import File
from image_app.models import MyImage
import cv2
import numpy as np

class imageSerializer(ModelSerializer):

   def create(self, validated_data):
      image = validated_data['model_pic'].read()
      img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
      img_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      #
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      #
      cv2.imwrite("image_app/image_after_predict/bang.jpg",img_gray)
      pic = MyImage()
      pic.model_pic.save(validated_data['model_pic'].name, File(open("image_app/image_after_predict/bang.jpg",'rb')))
      return pic

   class Meta:
      model = MyImage
      fields = [
         'model_pic'
      ]
