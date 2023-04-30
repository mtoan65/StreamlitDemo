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

model_path = 'checkpoint/'
if torch.cuda.is_available():
    map_location=torch.device('cuda')
else:
    map_location=torch.device('cpu')
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)



# Load model
def load_model(model_path, map_location):
    model = torch.load(model_path, map_location=map_location)
    return model

# Predict
def predict_class(model, image):
    image = image.to('cuda')
    logits = model(image)
    pred = logits.max(1, keepdim=True)[1]
    return pred

# Preprocessing
def preprocessing_uploader(image, input_size=512):
    img = imageio.imread(image)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, 2)
    img = Image.fromarray(img, mode='RGB')
    img = transforms.Resize(input_size + 16)(img)  # old: 16
    img = transforms.CenterCrop(input_size)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    return img

app_mode = st.sidebar.selectbox('Chọn trang',['Thông tin chung','Thống kê về dữ liệu huấn luyện','Ứng dụng chẩn đoán']) #two pages
if app_mode=='Thông tin chung':
    st.title('Giới thiệu về thành viên')
    # st.markdown("""
    # <style>
    # .big-font {
    # font-size:35px !important;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    # st.markdown("""
    # <style>
    # .name {
    # font-size:25px !important;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    
    # st.markdown('<p class="big-font"> Học sinh thực hiện </p>', unsafe_allow_html=True)
    # st.markdown('<p class="name"> I. Trần Mạnh Dũng - 11A1 </p>', unsafe_allow_html=True)
    # dung_ava = Image.open(your_path + 'member/Dung.jpg')
    # st.image(dung_ava)
    # st.markdown('<p class="name"> II. Lê Vũ Anh Tin - 8A2 </p>', unsafe_allow_html=True)
    # tin_ava = Image.open(your_path + 'member/Tin.jpg')
    # st.image(tin_ava)
    
    # st.markdown('<p class="big-font"> Giáo viên hướng dẫn đề tài </p>', unsafe_allow_html=True)
    # st.markdown('<p class="name"> Lê Thúy Phương Như - Giáo viên Sinh Học </p>', unsafe_allow_html=True)
    # Nhu_ava = Image.open(your_path + 'member/GVHD_Nhu.jpg')
    # st.image(Nhu_ava)
elif app_mode=='Thống kê về dữ liệu huấn luyện': 
    st.title('Thống kê tổng quan về tập dữ liệu')
    
    # st.markdown("""
    # <style>
    # .big-font {
    # font-size:30px !important;
    # }
    # </style>
    # """, unsafe_allow_html=True)
    # st.markdown('<p class="big-font"> I. Thông tin về tập dữ liệu </p>', unsafe_allow_html=True)
    # st.caption('Tập dữ liệu COVID-QU-EX được các nhà nghiên cứu của Đại Học Qatar (Qatar Univerersity) thu thập, làm sạch và chuẩn bị. Tập dữ liệu bao gồm 33920 ảnh X-quang lồng ngực, trong đó bao gồm 11956 ảnh có ghi nhận mắc covid-19, 11263 ảnh có ghi nhận mắc viêm phổi không do covid và 10701 ảnh bình thường. ')
    # st.caption('Nội dung nghiên cứu khoa học và ứng dụng của nhóm được thiết kế dựa trên việc huấn luyện nhóm dữ liệu Lung Segmentation Data. Dữ liệu đã được tiền xử lý và thay đổi kích thước về 256 x 256. Thông tin chi tiết của tập dữ liệu có thể tìm được ở dưới đây: ')
    # st.caption('*"https://www.kaggle.com/datasets/anasmohammedtahir/covidqu"*')
    # covid_dataset = Image.open(your_path + 'stat_image/covid_dataset.png')
    # st.image(covid_dataset)
    # #Vẽ sample ảnh
    # st.text('1) Một vài mẫu của ảnh x-quang lồng ngực mắc covid-19.')
    # covid_sample = Image.open(your_path + 'stat_image/covid_sample.png')
    # st.image(covid_sample)
    
    # st.text('2) Một vài mẫu của ảnh x-quang lồng ngực mắc viêm phổi thông thường.')
    # non_covid_sample = Image.open(your_path + 'stat_image/non_covid_sample.png')
    # st.image(non_covid_sample)
    
    # st.text('3) Một vài mẫu của ảnh x-quang lồng ngực khỏe mạnh.')
    # normal_sample = Image.open(your_path + 'stat_image/normal_sample.png')
    # st.image(normal_sample)
    
    # st.text('4) Vùng quan trọng được mô hình học máy chú ý.')
    # gradcam = Image.open(your_path + 'stat_image/gradcam.png')
    # st.image(gradcam)
    
    # #Vẽ thống kê tập dữ liệu
    # st.markdown('<p class="big-font"> II. Thống kê về tập dữ liệu </p>', unsafe_allow_html=True)
    # st.caption('Nhìn chung, dữ liệu tương đối cân bằng ở 3 lớp, trên cả tập huấn luyện và tập kiểm thử với lần lượt là ')
    # st.text('1) Biểu đồ cột so sánh số lượng dữ liệu tập huấn luyện (Train dataset)')
    # train_info = Image.open(your_path + 'stat_image/train_info.png')
    # st.image(train_info)
    # st.text('2) Biểu đồ cột so sánh số lượng dữ liệu tập kiểm thử (Validation dataset)')
    # valid_info = Image.open(your_path + 'stat_image/valid_info.png')
    # st.image(valid_info)
    # st.text('3) Biểu đồ tròn so sánh phần trăm dữ liệu tập huấn luyện (Train dataset)')
    # train_pie = Image.open(your_path + 'stat_image/train_pie.png')
    # st.image(train_pie)
    # st.text('4) Biểu đồ tròn so sánh phần trăm dữ liệu tập kiểm thử (Validation dataset)')
    # valid_pie = Image.open(your_path + 'stat_image/valid_pie.png')
    # st.image(valid_pie)
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
        
        # Preprocessing & Predict
        pred_resnet18_density = preprocessing_uploader(file, resnet18_density)
        pred_resnet18_birads = preprocessing_uploader(file, resnet18_birads)
        pred_resnet34_density = preprocessing_uploader(file, resnet34_density)
        pred_resnet34_birads = preprocessing_uploader(file, resnet34_birads)

        density_class = [0,1,2,3,4,5]
        birads_class = [0,1,2,3,4,5]

        result_resnet18_density = density_class[np.argmax(pred_resnet18_density)]
        result_resnet18_birads = birads_class[np.argmax(pred_resnet18_birads)]
        result_resnet34_density = density_class[np.argmax(pred_resnet34_density)]
        result_resnet34_birads = birads_class[np.argmax(pred_resnet34_birads)]

        # Display input image
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width = 400)

        # Display result
        st.write('- **Class density resnet18**: *{}%*'.format(round(result_resnet18_density[0,0] *100,2)))
        st.write('- **Class birads resnet18**: *{}%*'.format(round(result_resnet18_birads[0,0] *100,2)))
        st.write('- **Class density resnet34**: *{}%*'.format(round(result_resnet34_density[0,0] *100,2)))
        st.write('- **Class birads resnet34**: *{}%*'.format(round(result_resnet34_birads[0,0] *100,2)))
        
#         if str(result) == 'covid':
#             statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân mắc Covid-19.**')
#             st.error(statement)
#         elif str(result) == 'non-covid':
#             statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân mắc viêm phổi không do virus Covid-19 gây ra.**')
#             st.warning(statement)
#         elif str(result) == 'normal':
#             statement = str('Chẩn đoán của mô hình học máy: **Không có dấu hiệu bệnh viêm phổi.**')
#             st.success(statement)
        slot.success('Hoàn tất!')

# #         st.success(output)
     
#         #Plot bar chart
#         bar_frame = pd.DataFrame({'Xác suất dự đoán': [pred[0,0] *100, pred[0,1]*100, pred[0,2]*100], 
#                                    'Loại chẩn đoán': ["Covid-19", "Viêm phổi khác", "Bình thường"]
#                                  })
#         bar_chart = alt.Chart(bar_frame).mark_bar().encode(y = 'Xác suất dự đoán', x = 'Loại chẩn đoán' )
#         st.altair_chart(bar_chart, use_container_width = True)
#         #Note
#         st.write('- **Xác suất bệnh nhân mắc covid-19 là**: *{}%*'.format(round(pred[0,0] *100,2)))
#         st.write('- **Xác suất bệnh nhân mắc viêm phổi khác là**: *{}%*'.format(round(pred[0,1] *100,2)))
#         st.write('- **Xác suất bệnh nhân khỏe mạnh là**: *{}%*'.format(round(pred[0,2] *100,2)))
