"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import sys

def _main():
    log_dir         = sys.argv[1]#'model/'
    annotation_path = sys.argv[2]#'model_data/train.txt'
    classes_path    = sys.argv[3]#'model_data/voc_classes.txt'
    anchors_path    = sys.argv[4]#'model_data/yolo_anchors.txt'
    valSplit        = float(sys.argv[5])#0.2 #20% validation
    monitor         = 'val_loss'
    epoch           = int(sys.argv[6])#100
    batchSize       = int(sys.argv[7])#4
    stepMultiple    = int(sys.argv[8])#1
    getRandomData   = True 
    input_shape     = (416,416) # multiple of 32, hw
    class_names     = get_classes(classes_path)
    num_classes     = len(class_names)
    anchors         = get_anchors(anchors_path)
    is_tiny_version = len(anchors)==6 # default setting
    
    # if is_tiny_version:
    #     model = create_tiny_model(input_shape, anchors, num_classes,
    #         freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    # else:
    #     model = create_model(input_shape, anchors, len(class_names) , 
    #         load_pretrained = False ,
    #         weights_path='model/epoch10000_博物館YOLOv3.h5')
    model = create_model(input_shape, anchors, len(class_names) , 
                        load_pretrained = False ,
                        weights_path='model/20200312100epochs_yolov3.h5')

    train(model=model,annotation_path=annotation_path,input_shape=input_shape,anchors=anchors,num_classes=num_classes,
        log_dir=log_dir,valSplit=valSplit,monitor=monitor,epoch=epoch,batchSize=batchSize,stepMultiple=stepMultiple,getRandomData=getRandomData)

def train(model=None, annotation_path=None, input_shape=None, anchors=None, num_classes=None,
        log_dir='logs/',valSplit=None,monitor=None,epoch=None,batchSize=None,stepMultiple=None,getRandomData=True):
    
    model.compile(optimizer=Adam(lr=0.0001), 
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred},
                  metrics=['accuracy'])
        
    logging = TensorBoard(log_dir=log_dir,
                         histogram_freq=0,
        #                  batch_size=32, 
                         write_graph=True, 
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=0, 
                         embeddings_layer_names=None, 
                         embeddings_metadata=None)

    if(monitor == 'loss'):
        checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}.h5",
            monitor='loss', save_weights_only=True, save_best_only=True, period=1)
    else:
        checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    
    # reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=3, verbose=1)
    # early_stopping = EarlyStopping(monitor=monitor, min_delta=0, patience=10, verbose=1)

    callbacks_list = [logging,checkpoint]
    batch_size = batchSize
    val_split = valSplit
    epochs = epoch
    with open(annotation_path) as f:
        lines = f.readlines()
        print(len(lines))
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    history = model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes, getRandomData),
            steps_per_epoch=max(1, num_train//batch_size)*stepMultiple,
            validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors, num_classes, getRandomData),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs,
            initial_epoch=0,
            callbacks=callbacks_list)
    model.save_weights(log_dir + 'trained_weights.h5')

    #logging = TensorBoard(log_dir=log_dir)
    #checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    #    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines)*val_split)
    # num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    
    # if True:
    #     model.compile(optimizer=Adam(lr=1e-3), loss={
    #         # use custom yolo_loss Lambda layer.
    #         'yolo_loss': lambda y_true, y_pred: y_pred})

    #     batch_size = 32
    #     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #     model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    #             steps_per_epoch=max(1, num_train//batch_size),
    #             validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    #             validation_steps=max(1, num_val//batch_size),
    #             epochs=50,
    #             initial_epoch=0,
    #             callbacks=[logging, checkpoint])
    #     model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.

    # if True:
    #     for i in range(len(model.layers)):
    #         model.layers[i].trainable = True
    #     model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    #     print('Unfreeze all of the layers.')

    #     batch_size = 32 # note that more GPU memory is required after unfreezing the body
    #     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #     model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    #         steps_per_epoch=max(1, num_train//batch_size),
    #         validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    #         validation_steps=max(1, num_val//batch_size),
    #         epochs=100,
    #         initial_epoch=50,
    #         callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    #     model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


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


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    #model_body = multi_gpu_model(model_body,gpus=2)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        #原作者的部分
        # if freeze_body in [1, 2]:
        #     # Freeze darknet53 body or freeze all but 3 output layers.
        #     num = (185, len(model_body.layers)-3)[freeze_body-1]
        #     for i in range(num): model_body.layers[i].trainable = False
        #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        #原作者的部分
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers)-7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    model.summary()
    
    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        #原作者的部分
        # if freeze_body in [1, 2]:
        #     # Freeze the darknet body or freeze all but 2 output layers.
        #     num = (20, len(model_body.layers)-2)[freeze_body-1]
        #     for i in range(num): model_body.layers[i].trainable = False
        #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        #原作者的部分
        #調整的部分
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers)-7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        #調整的部分
    # 原作者部分
    # model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    #     arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
    #     [*model_body.output, *y_true])
    #調整的部分
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    #調整的部分
    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,getRandomData):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    #調整的部分
    #np.random.shuffle(annotation_lines)
    #調整的部分
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=getRandomData)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes, getRandomData):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, getRandomData)

if __name__ == '__main__':
    _main()
