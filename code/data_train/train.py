from ultralytics import YOLO


def main():
    # # 原始的数据路径    
    train_model_config_path = r'./config/train_model_config.yaml'
    # 3. 数据训练
    model = YOLO("yolo11s.yaml").load(r"./yolo11s.pt")  # build from YAML and transfer weights
    model.train(data=train_model_config_path, batch=16, epochs=300, imgsz=1280, device=[0,1])

    # 3. 模型转换
    path = model.export(format='onnx', half=True, imgsz=1280, device=0)
    print(path)


if __name__ == '__main__':
    main()
