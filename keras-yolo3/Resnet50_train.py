import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,LearningRateScheduler
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import os
import json

__data_dir          = "D:/JungWei/revisedYOLOv3_GithubDesktop/revisedYOLOv3/keras-yolo3/ResNetDataSet"
__train_dir         = os.path.join(__data_dir, "train")
__test_dir          = os.path.join(__data_dir, "test")
__trained_model_dir = os.path.join(__data_dir, "models")
__model_class_dir   = os.path.join(__data_dir, "json")
num_classes         = 4
__num_epochs        = 50
height_shift_range  = 0#[0, 0.25, 0.5, 0.75, 1]
width_shift_range   = 0#[0, 0.25, 0.5, 0.75, 1]
rotation_range      = 0 #[0, 15, 30]
zoom_range          = 0#[0, 0.15, 0.3, 0.45]
shear_range         = 0#[0, 0.15, 0.3]
training_image_size = 224
log_dir = 'D:/JungWei/revisedYOLOv3_GithubDesktop/revisedYOLOv3/keras-yolo3/logs/ResNet/'

def _main():
    # annotation_path = 'model_data/train.txt'
    # classes_path = 'model_data/voc_classes.txt'
    # anchors_path = 'model_data/yolo_anchors.txt'
    # valSplit = 0.1 #10% validation
    # monitor = 'val_loss'
    # epoch = 100
    # batchSize = 4
    # stepMultiple = 1
    # getRandomData = True 
    # input_shape = (416,416) # multiple of 32, hw
    # class_names = get_classes(classes_path)
    # num_classes = len(class_names)
    # anchors = get_anchors(anchors_path)
    # is_tiny_version = len(anchors)==6 # default setting
    
    # if is_tiny_version:
    #     model = create_tiny_model(input_shape, anchors, num_classes,
    #         freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    # else:
    #     model = create_model(input_shape, anchors, len(class_names) , 
    #         load_pretrained = False ,
    #         weights_path='model/epoch10000_博物館YOLOv3.h5')
    # model = create_model(input_shape, anchors, len(class_names) , 
    #                     load_pretrained = False ,
    #                     weights_path='model/epoch10000_博物館YOLOv3.h5')

    train(num_objects=num_classes, num_experiments=__num_epochs, enhance_data=False, 
             batch_size=8, show_network_summary=False, training_image_size = training_image_size, rotation_range=rotation_range,
             height_shift_range=height_shift_range, width_shift_range=width_shift_range, 
             zoom_range=zoom_range,shear_range=shear_range)

def train(num_objects=4, num_experiments=150, enhance_data=False, batch_size = 32, 
                   initial_learning_rate=1e-3, show_network_summary=False, training_image_size = 224,
                   rotation_range=0, height_shift_range=0.5, width_shift_range=0.5, zoom_range=0.3,
                   shear_range=0.2, optimizer='adam'):
    #lr_scheduler = LearningRateScheduler(self.lr_schedule)

    image_input = Input(shape=(training_image_size, training_image_size, 3))
    model = ResNet50(include_top=True,weights=None,classes=num_classes, input_tensor=image_input)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}.h5", monitor='val_acc', mode = 'max', verbose=1,
                                 save_weights_only=True, save_best_only=True)        

    if (enhance_data == True):
        print("Using Enhanced Data Generation")


    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        horizontal_flip=enhance_data, 
        height_shift_range=height_shift_range, 
        width_shift_range=width_shift_range,
        zoom_range=zoom_range,
        shear_range=shear_range)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(__train_dir, target_size=(training_image_size, training_image_size),
                                                        batch_size=batch_size,
                                                        class_mode="categorical")

    test_generator = test_datagen.flow_from_directory(__test_dir, target_size=(training_image_size, training_image_size),
                                                      batch_size=batch_size,
                                                      class_mode="categorical")

    class_indices = train_generator.class_indices
    class_json = {}
    for eachClass in class_indices:
        class_json[str(class_indices[eachClass])] = eachClass

    with open(os.path.join(__model_class_dir, "model_class.json"), "w+") as json_file:
        json.dump(class_json, json_file, indent=4, separators=(",", " : "),
                  ensure_ascii=True)
        json_file.close()
    print("JSON Mapping for the model classes saved to ", os.path.join(__model_class_dir, "model_class.json"))

    num_train = len(train_generator.filenames)
    num_test = len(test_generator.filenames)
    print("Number of experiments (Epochs) : ", __num_epochs)
            
    #early_stopping = EarlyStopping(monitor='val_acc', patience=200, verbose=2)

    history = model.fit_generator(train_generator, steps_per_epoch=int(num_train / batch_size), 
                                  epochs=__num_epochs,
                                  validation_data=test_generator,
                                  validation_steps=int(num_test / 6), 
                                  callbacks=[checkpoint])

    # model.save_weights(log_dir + 'trained_weights.h5')


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    _main()
