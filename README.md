# Hướng dẫn và Mô tả Dự án Emotion Detection

## 1. Mô tả về dataset
Dataset được sử dụng trong dự án này là FER2013, được thu thập và công bố trên trang web Kaggle bởi người dùng có tên là "deadskull7". FER2013 là một tập dữ liệu lớn chứa các hình ảnh khuôn mặt được gán nhãn với các loại cảm xúc khác nhau.

- **Nguồn gốc**: Tập dữ liệu FER2013 được thu thập từ nhiều nguồn khác nhau và bao gồm 35887 hình ảnh khuôn mặt có kích thước 48x48 pixels.
- **Loại hình ảnh**: Các hình ảnh trong tập dữ liệu này là ảnh màu xám (grayscale), có kích thước nhỏ và chứa các khuôn mặt của con người trong nhiều trạng thái cảm xúc khác nhau.
- **Các loại cảm xúc**: Tập dữ liệu này chứa bảy loại cảm xúc khác nhau, bao gồm: angry (tức giận), disgusted (kinh tởm), fearful (sợ hãi), happy (hạnh phúc), neutral (trung tính), sad (buồn) và surprised (ngạc nhiên).
- **Phân chia dữ liệu**: Tập dữ liệu FER2013 đã được chia thành các tập huấn luyện và kiểm tra, giúp cho việc đánh giá và kiểm tra hiệu suất của mô hình dễ dàng hơn.
- **Download tại**: [fer2013 (kaggle.com)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## 2. Cách chạy chương trình:

- Thực hiện các lệnh sau để tạo môi trường ảo cho dự án:
$ virtualenv emotion-detection-cnn
$ emotion-detection-cnn/Scripts/activate

- Install các thư viện cần thiết vào môi trường ảo:
$ pip install -r requirement.txt

- Di chuyển vào thư mục src:
$ cd ./src

- Training model:
$ python emotions.py --mode train
Lúc này chương trình sẽ thực hiện đọc dữ liệu từ dataset có trong folder data/train và training và tạo ra file model.h5.

- Chạy giao diện nhận diện cảm xúc:
$ python emotions.py --mode display