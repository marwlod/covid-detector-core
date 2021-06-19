from django.db import models


class Photo(models.Model):
    title = models.CharField(max_length=100)
    img = models.ImageField(upload_to='img/')
