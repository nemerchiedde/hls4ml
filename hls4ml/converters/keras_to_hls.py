from __future__ import print_function
import numpy as np
import h5py
import json
import math
from hls4ml.utils import plot_model
from hls4ml.model import HLSModel
#from hls4ml.writer.quartus_writer import write_activation_lstm

MAXMULT = 4096

class KerasFileReader(object):
    def __init__(self, config):
        self.config = config
        self.h5file = h5py.File(config['KerasH5'], mode='r')

    def __del__(self):
        if self.h5file:
            self.h5file.close()

    def _find_data(self, layer_name, var_name):
        def h5_visitor_func(name):
            if var_name in name:
                return name

        if 'model_weights' in list(self.h5file.keys()): # h5 file comes from model.save()
            layer_path = 'model_weights/{}'.format(layer_name)
        else:
            layer_path = layer_name

        data_path = self.h5file[layer_path].visit(h5_visitor_func)
        if data_path:
            return self.h5file['/{}/{}'.format(layer_path, data_path)]
        else:
            return None

    def get_weights_data(self, layer_name, var_name):
        data = self._find_data(layer_name, var_name)
        if data:
            return data[()]
        else:
            return None

    def get_weights_shape(self, layer_name, var_name):
        data = self._find_data(layer_name, var_name)
        if data is not None:
            return data.shape
        else:
            return None

class KerasModelReader(object):
    def __init__(self, keras_model):
        self.model = keras_model

    def get_weights_data(self, layer_name, var_name):
        layer = self.model.get_layer(layer_name)
        for i, w in enumerate(layer.weights):
            if var_name in w.name:
                try:
                    return w.numpy() # TF 2.x
                except:
                    return layer.get_weights()[i] # TF 1.x

        return None

    def get_weights_shape(self, layer_name, var_name):
        layer = self.model.get_layer(layer_name)
        for w in layer.weights:
            if var_name in w.name:
                return w.shape.as_list()

        return None

def get_qkeras_quantization(layer, keras_layer):
    if not layer['class_name'].startswith('Q'): # Not a QKeras layer, nothing to do
        return
    kernel_quantizer = keras_layer['config']['kernel_quantizer']['class_name']
    bias_quantizer = keras_layer['config']['bias_quantizer']['class_name']

    if kernel_quantizer != bias_quantizer:
        raise Exception('Mixing quantizers within QKeras layers is not supported')
    if kernel_quantizer == 'binary':
        layer['quantize'] = 2
    elif kernel_quantizer == 'ternary':
        layer['quantize'] = 3
    else:
        raise Exception('Unsupported quantizer {} in {} layer {}'.format(kernel_quantizer, layer['class_name'], layer['name']))


layer_handlers = {}

def register_keras_layer_handler(layer_name, handler_func):
    #print('keras_to_hls(93) - layer_name: ',layer_name, 'handler_func: ',handler_func)
    if layer_name in layer_handlers:
        raise Exception('Layer {} already registered'.format(layer_name))
    else:
        layer_handlers[layer_name] = handler_func

def get_supported_keras_layers():
    #print(list(layer_handlers.keys()))  ##['Conv1D', 'Conv2D', 'InputLayer', 'Reshape', 'Dense', 'BinaryDense', 'TernaryDense', 'Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'BatchNormalization', 'LSTM', 'Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate', 'MaxPooling1D', 'MaxPooling2D', 'AveragePooling1D', 'AveragePooling2D']
    return list(layer_handlers.keys())

def keras_handler(*args):
    def decorator(function):
        function.handles = [arg for arg in args]
        return function
    return decorator

def parse_default_keras_layer(keras_layer, input_names):
    layer = {}

    #Extract name for finding weights and biases
    layer['name'] = keras_layer['config']['name']
    layer['class_name'] = keras_layer['class_name']
    if input_names is not None:
        layer['inputs'] = input_names

    if 'activation' in keras_layer['config']:
        layer['activation'] = keras_layer['config']['activation']
        

    if 'epsilon' in keras_layer['config']:
        layer['epsilon'] = keras_layer['config']['epsilon']

    return layer

activation_type = []

def keras_to_hls(config): #__init__.py et a config vem do yml

    ######################
    ##  Do translation
    ######################

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []
    #print ('keras_to_hls',config)
    if 'KerasModel' in config:
        # Model instance passed in config from API
        model_arch = json.loads(config['KerasModel'].to_json())
        reader = KerasModelReader(config['KerasModel'])
    elif 'KerasJson' in config:                                              ##Usamos este
        # Extract model architecture from json
        with open(config['KerasJson']) as json_file:
            model_arch = json.load(json_file)
        reader = KerasFileReader(config)
        #print("reader",reader)
    elif 'KerasH5' in config:
        #print("terceiro")
        # Model arch and weights are in H5 file (from model.save() function)
        with h5py.File(config['KerasH5'], mode='r') as h5file:
            # Load the configuration from h5 using json's decode
            model_arch = h5file.attrs.get('model_config')
            if model_arch is None:
                raise ValueError('No model found in config file.')
            else:
                model_arch = json.loads(model_arch.decode('utf-8'))
        reader = KerasFileReader(config)
    else:
        raise ValueError('No model found in config file.')

    #print('JSON', model_arch)


    #Define layers to skip for conversion to HLS
    skip_layers = ['Dropout', 'Flatten']
    #All supported layers
    supported_layers = get_supported_keras_layers() + skip_layers

    #Map inputs of skipped and split (activation) layers
    inputs_map = {}

    #Loop through layers
    layer_counter = 0

    input_layers = None
    output_layers = None

    layer_config = None
    #layer_activation = []

    if model_arch['class_name'] == 'Sequential':  #model_arch contem o json   #Entra!
        print('Interpreting Sequential')
        layer_config = model_arch['config']
        #print("Layer Config JSON",layer_config)
        if 'layers' in layer_config: # Newer Keras versions have 'layers' in 'config' key   #Nao entra!
            layer_config = layer_config['layers']
        if layer_config[0]['class_name'] != 'InputLayer':
            input_layer = {}
            input_layer['name'] = 'input1'
            input_layer['class_name'] = 'InputLayer'
            input_layer['input_shape'] = layer_config[0]['config']['batch_input_shape'][1:]
            layer_list.append(input_layer)
            print('Input shape:', input_layer['input_shape'])
    elif model_arch['class_name'] in ['Model', 'Functional']:
        print('Interpreting Model')
        layer_config = model_arch['config']['layers']
        input_layers = [ inp[0] for inp in model_arch['config']['input_layers'] ]
        output_layers = [ out[0] for out in model_arch['config']['output_layers'] ]

    # Get input shape and check for unsupported layer type
    for keras_layer in layer_config: #layer_config é model_arch['config']
        if keras_layer['class_name'] not in supported_layers:
            raise Exception('ERROR: Unsupported layer type: {}'.format(keras_layer['class_name']))

    output_shapes = {}
    output_shape = None

    print('Topology:')
    for keras_layer in layer_config:
        if 'batch_input_shape' in keras_layer['config']:
            input_shapes = [keras_layer['config']['batch_input_shape']] #Recupera input_shape = [[None, 5, 1]]
            print('keras_to_hls(215) - input_shapes: ',input_shapes, "keras_layer['config']: ",keras_layer['config'])
            if 'recurrent_activation' in keras_layer['config']:
                layer_rec_activation = keras_layer['config']['recurrent_activation']  #Recupera recurrent_activation da LSTM
                return_sequences = keras_layer['config']['return_sequences']
                sliding_window = keras_layer['config']['sliding_window']
                #print('rec_activation',layer_rec_activation)
                print('keras_to_hls(220) - return_sequences: ',return_sequences)
        else:
            if 'inbound_nodes' in keras_layer:                  #Nao entra
                input_shapes = [output_shapes[inbound_node[0][0]] for inbound_node in keras_layer['inbound_nodes']]
            else:
                # Sequential model, so output_shape from the previous layer is still valid
                input_shapes = [output_shape]
                #print("Input_shapes_Topology",input_shapes)  #[[None, 5, 1]]

        keras_class = keras_layer['class_name'] #InputLayer, LSTM, Dense

        if keras_class in skip_layers:      #Nao entra
            if 'inbound_nodes' in keras_layer:
                name = keras_layer['config']['name']
                #Currently supported skipped layers have only one input
                parent_input = keras_layer['inbound_nodes'][0][0][0]
                #Skipped layers can follow each other (e.g., Dropout -> Flatten)
                inputs_map[name] = inputs_map.get(parent_input, parent_input)

            if keras_class == 'Flatten':
                output_shapes[keras_layer['config']['name']] = [input_shapes[0][0], np.prod(input_shapes[0][1:])]
            else:
                output_shapes[keras_layer['config']['name']] = input_shapes[0]

            continue

        if keras_class in supported_layers:
            layer_counter = layer_counter + 1
        #print("layer_counter", layer_counter)

        #Extract inbound nodes
        if 'inbound_nodes' in keras_layer and len(keras_layer['inbound_nodes']) > 0:   #Nao entra
            input_names = [ inputs_map.get(inp[0], inp[0]) for inp in keras_layer['inbound_nodes'][0] ]
        else:
            input_names = None
        #print("keras_layer", keras_layer, "input_names", input_names,"input_shapes", input_shapes, "reader" ,  reader, "config", config)



        #if keras_layer['class_name'] =='LSTM':
        #    layer, output_shape = layer_handlers[keras_class](keras_layer, input_names, input_shapes, reader, config, return_sequences)
        #else:
        #    layer, output_shape = layer_handlers[keras_class](keras_layer, input_names, input_shapes, reader, config)


        layer, output_shape = layer_handlers[keras_class](keras_layer, input_names, input_shapes, reader, config)
        #print('layer_handlers',layer_handlers[keras_class])
        #print('Layer name: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
        layer_list.append( layer )
        #print('layer',layer)
        #print('output_shape',output_shape) #[None, 5, 1] lstm, [None, 1] dense (errado)
        #print("O que é 0:", layer['class_name'])
        if 'activation' in layer and layer['class_name'] not in ['Activation', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'LSTM']:# + qkeras_layers: #Nao entra a LSTM, somente o DENSE
            act_layer = {}
            #print("O que é Dense1:",layer['class_name'] )   #Dense
            act_layer['name'] = layer['name'] + '_' + layer['activation']
            act_layer['activation'] = layer['activation']
            #print('O que é Dense2:',act_layer['activation']) #Relu
            #print('O que é Desne3:',act_layer) #{'name': 'dense_relu', 'activation': 'relu'}
            if 'activ_param' in layer:  #Nao entra
                act_layer['activ_param'] = layer['activ_param']
                act_layer['class_name'] = layer['activation']
            elif layer['activation'] == 'softmax':
                act_layer['class_name'] = 'Softmax'
            else:
                act_layer['class_name'] = 'Activation'
            inputs_map[layer['name']] = act_layer['name']
            if output_layers is not None and layer['name'] in output_layers:
                output_layers = [act_layer['name'] if name == layer['name'] else name for name in output_layers]

            #print('O que é 0',act_layer)
            layer_list.append(act_layer)
            #print('O que é layer list',layer_list)

            #print('nemer chiedde')
            #print('layer list 1', layer_list[1]['activation']) #recurrent_activation

            #layer_activation.append(layer_list[1]['activation'])
            #layer_activation.append(layer_list[2]['activation'])
            #print(layer_activation)
            #print (layer_rec_activation)
            activation_type.append(layer_list[1]['activation'])
            activation_type.append(layer_rec_activation)



        assert(output_shape is not None)
        #print("output_shape 0", output_shape)
        output_shapes[layer['name']] = output_shape
        print("keras_to_hls(331) - output_shape:", output_shapes[layer['name']], layer['name'])


    #################
    ## Generate HLS
    #################
    ## Read file lstm_cell.h
    #print( "activation",activation_type)

    #write_activation_lstm(layer_rec_activation,layer_activation)
    #modification_activation_lstm(layer_rec_activation,layer_activation)

    print('Creating HLS model')
    print("keras_to_hls(326) - layer_list: ", layer_list)
    print("keras_to_hls(325) - output_layers: ", output_layers)
    print('keras_to_hls(326) - input_layer: ', input_layers)
    hls_model = HLSModel(config, reader, activation_type, return_sequences, sliding_window, layer_list, input_layers, output_layers)
    plot = plot_model(hls_model)
    print('HLS_MODEL: ', hls_model)
    print('**Input_Layers**', input_layers)
    print('**Output_Layers**', output_layers)
    return hls_model
