import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
import altair as alt
from PIL import Image, ImageOps
from io import BytesIO
import imageio
from torchvision import transforms
from models.Resnet18 import resnet18
from models.Resnet34 import resnet34

model_path = 'checkpoint/'
if torch.cuda.is_available():
    map_location=torch.device('cuda')
    device = 'cuda'
else:
    map_location=torch.device('cpu')
    device = 'cpu'
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)



# Load model
def load_model(model_path, map_location):
    if "resnet18" in model_path:
        model = resnet18(False,'', device)
    else:
        model = resnet34(False,'', device)
    checkpoint = torch.load(model_path, map_location=map_location)['model_state_dict']
    print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    return model

# Predict
def predict_class(model, image):
    image = image.to(device)
    logit = model(image)
    pred = logit.max(1, keepdim=True)[1]
    return torch.flatten(pred)[0]

# Preprocessing
def preprocessing_uploader(image_file, input_size=512):
    bytes_data = image_file.getvalue()
    inputShape = (512, 512)
    img = Image.open(BytesIO(bytes_data))
    img = img.convert("RGB")
    img = img.resize(inputShape)
    T = transforms.Resize(input_size + 16)
    img = T(img)  # old: 16
    img = transforms.CenterCrop(input_size)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = torch.unsqueeze(img, dim=0)
    return img

mode = ['Thông tin chung','Thống kê về dữ liệu huấn luyện','Ứng dụng chẩn đoán']
app_mode = st.sidebar.selectbox('Chọn trang',mode) #two pages
if app_mode=='Thông tin chung':
    st.title('Giới thiệu về thành viên')
elif app_mode=='Thống kê về dữ liệu huấn luyện': 
    st.title('Thống kê tổng quan về tập dữ liệu')
elif app_mode=='Ứng dụng chẩn đoán':
    resnet18_density = load_model(model_path + 'resnet18_density/best_model.pth', map_location)
    resnet34_density = load_model(model_path + 'resnet34_density/best_model.pth', map_location)
    resnet18_birads = load_model(model_path + 'resnet18_birads/best_model.pth', map_location)
    resnet34_birads = load_model(model_path + 'resnet34_birads/best_model.pth', map_location)
    st.title('Ứng dụng chẩn đoán trong nhũ ảnh X-quang')

    file = st.file_uploader("Bạn vui lòng nhập ảnh x-quang để phân loại ở đây", type=["jpg", "png"])
# 

    if file is None:
        st.text('Vui lòng chọn file khác....')

    else:
        slot = st.empty()
        slot.text('Hệ thống đang thực thi chẩn đoán....')
        
        # Preprocessing 
        image = preprocessing_uploader(file)

        # Predict
        pred_resnet18_density = predict_class(resnet18_density, image) 
        pred_resnet18_birads = predict_class(resnet18_birads, image) 
        pred_resnet34_density = predict_class(resnet34_density, image) 
        pred_resnet34_birads = predict_class(resnet34_birads, image) 

        density_class = [2,3,4]
        birads_class = [1,2,3]

        # result_resnet18_density = density_class[np.argmax(pred_resnet18_density)]
        # result_resnet18_birads = birads_class[np.argmax(pred_resnet18_birads)]
        # result_resnet34_density = density_class[np.argmax(pred_resnet34_density)]
        # result_resnet34_birads = birads_class[np.argmax(pred_resnet34_birads)]


        # Display input image
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width = 400)

        # Display result
        st.write('- **Class density resnet18**: *{}*'.format(pred_resnet18_density))
        st.write('- **Class birads resnet18**: *{}*'.format(pred_resnet18_birads))
        st.write('- **Class density resnet34**: *{}*'.format(pred_resnet34_density))
        st.write('- **Class birads resnet34**: *{}*'.format(pred_resnet34_birads))
        
        slot.success('Hoàn tất!')