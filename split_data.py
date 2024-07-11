import os
import random
import shutil

def split_data(data_path, splited_data_path, train_size, val_data = True):
    data = os.listdir(data_path)

    data_len = int(len(data)/2)
    train_len = int(data_len * train_size)
    test_len = int((data_len - train_len)/2)
    val_len = int(data_len - train_len - test_len)

    train_data = []
    test_data = []
    val_data = []

    for i in range(train_len):
        x = random.choice(data)
        train_data.append(x)
        data.remove(x)
        if x[-4:] == ".xml":
            for type in [".jpeg", ".jpg", ".png"]:
                if x[:-4] + type in data:
                    train_data.append(x[:-4] + type)
                    data.remove(x[:-4] + type)
        elif x[-4:] in [".jpg", ".png"]:
            for type in [".xml"]:
                if x[:-4] + type in data:
                    train_data.append(x[:-4] + type)
                    data.remove(x[:-4] + type)
        elif x[-5:] in [".jpeg"]:
            for type in [".xml"]:
                if x[:-5] + type in data:
                    train_data.append(x[:-5] + type)
                    data.remove(x[:-5] + type)
        else:
            print(f"{x} Error!")
        if len(train_data) % 2 != 0:
            print(f"Train data: {len(train_data)}")
        
    for i in range(test_len):
        x = random.choice(data)
        test_data.append(x)
        data.remove(x)
        if x[-4:] == ".xml":
            for type in [".jpeg", ".jpg", ".png"]:
                if x[:-4] + type in data:
                    test_data.append(x[:-4] + type)
                    data.remove(x[:-4] + type)
        elif x[-4:] in [".jpg", ".png"]:
            for type in [".xml"]:
                if x[:-4] + type in data:
                    test_data.append(x[:-4] + type)
                    data.remove(x[:-4] + type)
        elif x[-5:] in [".jpeg"]:
            for type in [".xml"]:
                if x[:-5] + type in data:
                    test_data.append(x[:-5] + type)
                    data.remove(x[:-5] + type)
        else:
            print(f"{x} Error!")
        if len(test_data) % 2 != 0:
            print(f"Test data: {len(test_data)}")


    for i in range(val_len):
        x = random.choice(data)
        val_data.append(x)
        data.remove(x)
        if x[-4:] == ".xml":
            for type in [".jpeg", ".jpg", ".png"]:
                if x[:-4] + type in data:
                    val_data.append(x[:-4] + type)
                    data.remove(x[:-4] + type)
        elif x[-4:] in [".jpg", ".png"]:
            for type in [".xml"]:
                if x[:-4] + type in data:
                    val_data.append(x[:-4] + type)
                    data.remove(x[:-4] + type)
        elif x[-5:] in [".jpeg"]:
            for type in [".xml"]:
                if x[:-5] + type in data:
                    val_data.append(x[:-5] + type)
                    data.remove(x[:-5] + type)
        else:
            print(f"{x} Error!")
        if len(val_data) % 2 != 0:
            print(f"Val data: {len(val_data)}")

    os.makedirs(splited_data_path + "/train", exist_ok=True)
    os.makedirs(splited_data_path + "/test", exist_ok=True)
    os.makedirs(splited_data_path + "/val", exist_ok=True)

    for train in train_data:
        shutil.copyfile(data_path + "/" + train, splited_data_path + "/train/" + train)
    for test in test_data:
        shutil.copyfile(data_path + "/" + test, splited_data_path + "/test/" + test)
    for val in val_data:
        shutil.copyfile(data_path + "/" + val, splited_data_path + "/val/" + val)


    print(f"train: {len(train_data)/2} test: {len(test_data)/2} val: {len(val_data)/2}")    
    print("Done!")


data_path = "TensorFlow/workspace/training_demo/images/all_data"
splited_data_path = "TensorFlow/workspace/training_demo/images/new_data"
split_data(data_path, splited_data_path, 0.7)