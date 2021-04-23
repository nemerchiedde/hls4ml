from __future__ import print_function
import six
import os
import sys
import re
import numpy as np
import pandas as pd

from collections import OrderedDict

class Quantizer(object):
    def __init__(self, bits, hls_type):
        self.bits = bits
        self.hls_type = hls_type

    def __call__(self, data):
        raise NotImplementedError

class IntegerPrecisionType(object):
    def __init__(self, width=16, signed=True):
        self.width = width
        self.signed = signed

        #replaced __str__ by a parsing function in the backend

class FixedPrecisionType(object):
    def __init__(self, width=16, integer=6, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        self.width = width
        self.integer = integer
        self.signed = signed
        self.rounding_mode = rounding_mode
        self.saturation_mode = saturation_mode
        self.saturation_bits = saturation_bits

        #replaced __str__ by a parsing function in the backend

class HLSType(object):
    def __init__(self, name, precision, **kwargs):
        self.name = name.format(**kwargs)
        self.precision = precision

    def get_precision(self):
        return self.precision
    def set_precision(self, precision):
        self.precision = precision
    """def definition_cpp(self):
        return 'typedef {precision} {name};\n'.format(name=self.name, precision=self.precision)"""

class CompressedType(HLSType):
    def __init__(self, name, precision, index_precision, **kwargs):
        super(CompressedType, self).__init__('compressed_type{index}', precision, **kwargs)
        self.index_precision = index_precision

    """def definition_cpp(self):
        cpp_fmt = ('typedef struct {name} {{ '
               '{index} row_index; '
               '{index} col_index; '
               '{precision} weight; }} {name};\n')
        return cpp_fmt.format(name=self.name, index=self.index_precision, precision=self.precision)"""

class Variable(object):
    def __init__(self, var_name, type_name, precision, **kwargs):
        self.name = var_name.format(**kwargs)
        self.type = HLSType(type_name, precision, **kwargs)
        self.cppname = re.sub(r'\W|^(?=\d)','_', self.name)

class ArrayVariable(Variable):

    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, pragma='partition', **kwargs):
        super(ArrayVariable, self).__init__(var_name, type_name, precision, **kwargs)
        self.shape = shape
        self.dim_names = dim_names
        self.pragma = pragma

    def __str__(self):
        return 'ArrayVariable of type: {type}, name: {name} shape: {shape}'.format(type=self.type.name, name=self.cppname, shape=self.shape)

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def definition_cpp(self):
        array_shape = self.size_cpp()
        return '{type} {name}[{shape}]'.format(type=self.type.name, name=self.cppname, shape=array_shape)

    def definition_cpp_name(self):
        return '{name}'.format(name=self.name)

    def definition_cpp_type(self):
        return '{type}'.format(type=self.type.name)

    def size(self):
        nelem = 1
        for dim in self.shape:
            nelem *= dim
        return nelem

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

class InplaceVariable():
    def __init__(self, shape, dim_names, proxy, **kwargs):
        self.shape = shape
        self.dim_names = dim_names
        self.type = proxy.type
        self.name = proxy.name
        self.size = proxy.size

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def definition_cpp(self):
        return None

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

class WeightVariable(Variable):
    def __init__(self, var_name, type_name, precision, data, quantizer=None, **kwargs):
        super(WeightVariable, self).__init__(var_name, type_name, precision, **kwargs)
        self.data = data
        self.nzeros = -1
        self.shape = list(self.data.shape)
        self.data_length = np.prod(self.data.shape)
        self.nonzeros = np.count_nonzero(self.data)
        self.nzeros = self.data_length - self.nonzeros
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self._iterator = None
        self.update_precision(precision)
        self.quantizer = quantizer

    def __iter__(self):
        self._iterator = np.nditer(self.data, order='C')
        return self

    def __next__(self):
        if not self._iterator.finished:
            value = self._iterator[0]
            self._iterator.iternext()
            return self.precision_fmt % value
        else:
            raise StopIteration

    next = __next__

    def update_precision(self, new_precision):
        self.type.precision = new_precision
        precision_str = str(self.type.precision)
        if 'int' in precision_str:
            self.precision_fmt = '%d'
        else:
            match = re.search('.+<(.+?)>', precision_str)
            if match is not None:
                precision_bits = match.group(1).split(',')
                width_bits = int(precision_bits[0])
                integer_bits = int(precision_bits[1])
                fractional_bits = integer_bits - width_bits
                lsb = 2 ** fractional_bits
                if lsb < 1:
                    # Use str to represent the float with digits, get the length
                    # to right of decimal point
                    decimal_spaces = len(str(lsb).split('.')[1])
                else:
                    decimal_spaces = len(str(2**integer_bits))
                self.precision_fmt = '%.{}f'.format(decimal_spaces)
            else:
                self.precision_fmt = '%f'

    def definition_cpp(self):
        return '{type} {name}[{size}]'.format(type=self.type.name, name=self.cppname, size=self.data_length)

class CompressedWeightVariable(WeightVariable):
    def __init__(self, var_name, type_name, precision, data, reuse_factor, quantizer=None, **kwargs):
        super(CompressedWeightVariable, self).__init__(var_name, type_name, precision, data, quantizer=quantizer, **kwargs)
        self.extra_zeros = 0
        self.data_length = np.prod(data.shape) - self.nzeros
        while self.data_length % reuse_factor != 0:
            self.extra_zeros += 1
            self.data_length += 1
        self.nonzeros = np.prod(data.shape) - self.nzeros + self.extra_zeros

        # Compress the array
        weights = []
        extra_nzero_cnt = self.extra_zeros
        it = np.nditer(data, order='C', flags=['multi_index'])
        max_idx = 0
        while not it.finished:
            val = it[0]
            if not (val == 0 and extra_nzero_cnt < 1):
                if val == 0:
                    extra_nzero_cnt -= 1
                if it.multi_index[0] > max_idx:
                    max_idx = it.multi_index[0]
                if it.multi_index[1] > max_idx:
                    max_idx = it.multi_index[1]
                weights.append([it.multi_index[1], it.multi_index[0], val])
            it.iternext()
        weights.sort()

        index_precision = 32
        if max_idx > 0:
            index_precision = int(np.log2(max_idx) + 1)
        self.type = CompressedType(type_name, precision, IntegerPrecisionType(width=index_precision, signed=False), **kwargs)

        self.data = weights

    def __iter__(self):
        self._iterator = iter(self.data)
        return self

    def __next__(self):
        value = next(self._iterator)
        value_fmt = self.precision_fmt % value[2]
        return '{ %u, %u, %s }' % (value[1], value[0], value_fmt)

    next = __next__

class Layer(object):
    def __init__(self, model, name, attributes, inputs, outputs=None):
        self.model = model
        self.name = name
        self.index = model.next_layer()
        self.inputs = inputs
        self.outputs = outputs
        print('hls_layers(221) - Layer_output:', self.outputs)
        if self.outputs is None:
            self.outputs = [self.name]

        self.attributes = attributes
        print('hls_layers(226) - atributos:', attributes)

        self._function_template = self.model.config.backend.get_function_template(self.__class__.__name__)
        print("hls_layers(229) - self._function_template:", self._function_template)
        self._config_template = self.model.config.backend.get_config_template(self.__class__.__name__)
        print("hls_layers(230) - self._config_template:", self._config_template)
        self.include_list = self.model.config.backend.get_include_list(self.__class__.__name__)
        self.weights = OrderedDict()
        self.variables = OrderedDict()
        self.precision = OrderedDict()
        accum_t = HLSType(*reversed(self.model.config.get_precision(self, 'accum')))
        self.precision[accum_t.name] = accum_t
        self.set_attr('accum_t', accum_t.precision)
        self.reuse_factor = self.model.config.get_reuse_factor(self)

        layer_config = self.model.config.get_layer_config(self)
        print("hls_layers(241) - self.model.config.get_layer_config(self):", self.model.config.get_layer_config(self))
        for config_key, config_value in layer_config.items():
            if config_key in self.attributes:
                print('WARNING: Config parameter "{}" overwrites an existing attribute in layer "{}" ({})'.format(config_key, self.name, self.__class__.__name__))
            self.attributes[config_key] = config_value

        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def set_attr(self, key, value):
        self.attributes[key] = value

    def get_attr(self, key, default=None):
        return self.attributes.get(key, default)

    def get_input_node(self, input_name=None):
        if input_name is not None:
            print('hls_layers(260) - input_name:', input_name, self.model.graph.get(input_name))
            return self.model.graph.get(input_name)
        else:
            print('hls_layers(263) - input_name:', self.inputs, self.inputs[0], self.model.graph.get(self.inputs[0]))
            return self.model.graph.get(self.inputs[0])

    def get_input_variable(self, input_name=None):  #recupera
        if input_name is not None:
            print('hls_layers(266) - input_name:', input_name) #nao entra
            return self.model.get_layer_output_variable(input_name)
        else:
            print('hls_layers(269) - input_name:', input_name, 'inputd:', self.inputs[0])
            return self.model.get_layer_output_variable(self.inputs[0])

    def get_output_nodes(self, output_name=None):
        if output_name is None:
            output_name = self.outputs[0]
        return [node for node in self.model.graph.values() if node.inputs[0] == output_name]

    def get_output_variable(self, output_name=None):

        if output_name is not None:
            return self.variables[output_name]
        else:
            return next(iter(self.variables.values()))

    def get_weights(self, var_name=None):
        if var_name:
            return self.weights[var_name]

        return self.weights.values()

    def get_variables(self):
        return self.variables.values()
    #shape=[]


    def add_output_variable(self, shape, dim_names, out_name=None, var_name='layer{index}_out', type_name='layer{index}_t', precision=None, pragma='auto'):
        if out_name is None:
            out_name = self.outputs[0]
            print('hls_layers(295) - out_name:',out_name) #lstm_input, lstm, dense, dense_relu

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var='result')
            print('hls_layers(299) - precision:',precision) # ac_fixed<16, 6, true>
        if pragma == 'auto':
            if self.model.config.get_config_value('IOType') == 'io_serial':
                pragma = 'stream'
            else:
                if self.name in self.model.inputs:
                    pragma = 'reshape'
                else:
                    pragma = 'partition'
        print('hls_layers(308) - pragma:', pragma)  #1- reshape,  2,3,4- Partition
        out = ArrayVariable(shape, dim_names, var_name=var_name, type_name=type_name, precision=precision, pragma=pragma, index=self.index)
        #if out_name == 'lstm_input':
        #    shape = [1, 1]
        #    out = ArrayVariable(shape, dim_names, var_name=var_name, type_name=type_name, precision=precision, pragma=pragma, index=self.index)
        #else:
    #        out = ArrayVariable(shape, dim_names, var_name=var_name, type_name=type_name, precision=precision, pragma=pragma, index=self.index)
        print('hls_layers(310) - out:', out.shape, out.type.name, out.dim_names, out_name)
        self.variables[out_name] = out
        self.model.register_output_variable(out_name, out)

        self.precision[out.type.name] = out.type

    def add_weights(self, quantizer=None, compression=False):
        data = self.model.get_weights_data(self.name, 'kernel')

        self.add_weights_variable(name='weight', var_name='w{index}', data=data, quantizer=quantizer, compression=compression)

    def add_bias(self, quantizer=None):
        data = self.model.get_weights_data(self.name, 'bias')
        precision = None
        type_name = None
        if data is None:
            data = np.zeros(self.get_output_variable().shape[-1])
            precision = IntegerPrecisionType(width=1, signed=False)
            type_name = 'bias{index}_t'
            quantizer = None # Don't quantize non-existant bias

        self.add_weights_variable(name='bias', var_name='b{index}', type_name=type_name, precision=precision, data=data, quantizer=quantizer)

    def add_weights_variable(self, name, var_name=None, type_name=None, precision=None, data=None, quantizer=None, compression=False):
        if var_name is None:
            var_name = name + '{index}'

        if precision is None:
            precision, _ = self.model.config.get_precision(self, var=name)

        if type_name is None:
            _, type_name = self.model.config.get_precision(self, var=name)

        if data is None:
            data = self.model.get_weights_data(self.name, name)
        elif isinstance(data, six.string_types):
            data = self.model.get_weights_data(self.name, data)

        data_unquantized = data
        if quantizer is not None:
            precision = quantizer.hls_type
            type_name = name + '{index}_t'
            data = quantizer(data)

        if compression:
            var = CompressedWeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, reuse_factor=self.reuse_factor, index=self.index)
        else:
            var = WeightVariable(var_name, type_name=type_name, precision=precision, quantizer=quantizer, data=data, index=self.index)

            var.data_unquantized = data_unquantized
        self.weights[name] = var
        self.precision[var.type.name] = var.type

    def _default_function_params(self):
        params = {}
        params['config'] = 'config{}'.format(self.index)
        params['input_t'] = self.get_input_variable().type.name
        print('hls_layers(367) - params[input_t]:', params['input_t'])
        params['output_t'] = self.get_output_variable().type.name
        print('hls_layers(369) - params[output_t]:', params['output_t'])
        params['input'] = self.get_input_variable().name
        print('hls_layers(371) - params[input]:', params['input'])
        params['output'] = self.get_output_variable().name
        print('hls_layers(373) - params[output]:', params['output'])

        return params

    def _default_config_params(self):
        params = {}
        params.update(self.attributes)
        params['index'] = self.index
        params['iotype'] = self.model.config.get_config_value('IOType')
        params['reuse'] = self.reuse_factor

        # data types
        for weight_name, variable in self.weights.items():
            params[weight_name + '_t'] = variable.type.name

        return params

    def get_layer_precision(self):
        for obj in self.precision:
            self.precision[obj].set_precision(precision=self.get_precision_string(self.precision[obj].get_precision()))
        return self.precision

    def get_precision_string(self, precision):
        return self.model.config.backend.get_precision_string_backend(precision)

    def var_definition_cpp(self, var):
        if isinstance(var, CompressedType):
            cpp_fmt = ('typedef struct {name} {{'
                   '{index} row_index; '
                   '{index} col_index; '
                   '{precision} weight;}} {name};\n')
            typestring = cpp_fmt.format(name=var.name, index=self.get_precision_string(var.index_precision), precision=self.get_precision_string(var.precision))
        elif isinstance(var, HLSType):
            typestring = 'typedef {precision} {name};\n'.format(name=var.name, precision=self.get_precision_string(var.precision))
        else:
            typestring = var
        return typestring

    # myproject.cpp/h
    def function_cpp(self):
        raise NotImplementedError

    # parameters.h
    def config_cpp(self):
        raise NotImplementedError

    def get_numbers_cpp(self):
        numbers = ''
        for k, v in self.get_output_variable().get_shape():
            numbers += '#define {} {}\n'.format(k,v)

        return numbers

    def precision_cpp(self):
        return 'typedef {precision} layer{index}_t;'.format(precision=self.get_output_variable().precision, index=self.index)

class Input(Layer):
    def initialize(self):
        shape = self.attributes['input_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_INPUT_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]
        if self.index == 1:
            default_type_name = 'input_t'
        else:
            default_type_name = 'input{}_t'.format(self.index)
        type_name = self.attributes.get('type_name', default_type_name)
        precision = self.attributes.get('precision', None)
        self.add_output_variable(shape, dims, var_name=self.name, type_name=type_name, precision=precision)

    def function_cpp(self):
        return None

    def config_cpp(self):
        return None

class Reshape(Layer):
    def initialize(self):
        shape = self.attributes['target_shape']
        if shape[0] is None:
            shape = shape[1:]
        dims = ['N_SIZE_{}_{}'.format(i, self.index) for i in range(1, len(shape) + 1)]

        out_name = self.outputs[0]
        proxy = self.get_input_variable()
        out = InplaceVariable(shape, dims, proxy, index=self.get_input_node().index)

        self.variables[out_name] = out
        self.model.register_output_variable(out_name, out)

    def function_cpp(self):
        return None

    def config_cpp(self):
        return None

class Dense(Layer):
    def initialize(self):
        shape = [self.attributes['n_out']]
        dims = ['N_LAYER_{}'.format(self.index)]
        compression = self.model.config.get_compression(self)
        self.model.config.backend.set_strategy(self)
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'), compression=compression)
        index_t = IntegerPrecisionType(width=1, signed=False)

        self.model.config.backend.configure_weights(self)

        if self.model.config.is_resource_strategy(self):
            if self.model.config.get_compression(self):
                index_t = self.get_weights('weight').type.index_precision

        self.set_attr('index_t', index_t)
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))


    def function_cpp(self):
        params = self._default_function_params()
        params['strategy'] = self.get_attr('strategy')
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def binary_check(self):
        params = self._default_config_params()
        return self.get_layer_precision()[params['weight_t']].get_precision() == 'ac_int<2, true>' or self.get_layer_precision()[params['weight_t']].get_precision() == 'ap_int<2>'

    def ternary_check(self):
        params = self._default_config_params()
        return self.get_layer_precision()[params['weight_t']].get_precision() == 'ac_int<1, false>' or self.get_layer_precision()[params['weight_t']].get_precision() == 'ac_uint<1>'

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['n_out'] = self.get_output_variable().size_cpp()
        params['nzeros'] = self.get_weights('weight').nzeros
        params['nonzeros'] = self.get_weights('weight').nonzeros
        params['index_t'] = self.get_precision_string(self.get_attr('index_t'))
        params['accum_t'] = self.get_precision_string(self.get_attr('accum_t'))

        return self._config_template.format(**params)

class Conv1D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['n_out'], self.attributes['n_filt']]
            dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['n_out']]
            dims = ['N_FILT_{}'.format(self.index), 'N_OUTPUTS_{}'.format(self.index)]

        self.add_output_variable(shape, dims)
        self.add_weights(quantizer = self.get_attr('weight_quantizer'))
        self.add_bias(quantizer = self.get_attr('bias_quantizer'))
        if self.model.config.is_resource_strategy(self):
            self.set_attr('strategy', 'large')
            if self.model.config.backend.name == 'Vivado':
                self.model.config.backend.set_closest_reuse_factor(self)
                self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[2, 1, 0]) #(W,C,F) => (F,C,W)
        else:
            self.set_attr('strategy', 'latency')

    def function_cpp(self):
        params = self._default_function_params()
        params['strategy'] = self.get_attr('strategy')
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        input_dims = self.get_input_variable().dim_names
        if self.get_attr('data_format') == 'channels_last':
            params['n_in'] = '*'.join([str(k) for k in input_dims[:-1]])
            params['n_chan'] = input_dims[-1]
        else:
            params['n_in'] = '*'.join([str(k) for k in input_dims[1:]])
            params['n_chan'] = input_dims[0]
        params['dilation'] = self.get_attr('dilation', 1)
        params['n_filt'] = 'N_FILT_{}'.format(self.index)
        params['n_out'] = 'N_OUTPUTS_{}'.format(self.index)
        params['nzeros'] = self.get_weights('weight').nzeros
        params['config_t'] = 'std::nullptr_t'
        params['accum_t'] = self.get_precision_string(self.get_attr('accum_t'))

        if self.model.config.is_resource_strategy(self):
            params['config_t'] = 'config{}_mult'.format(self.index)
            conv_config = self._config_template[0].format(**params)

            mult_params = self._default_config_params()
            mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_width')
            mult_params['n_out'] = self.get_attr('n_filt')
            mult_config = self._config_template[1].format(**mult_params)

            return mult_config + '\n' + conv_config
        else:
            return self._config_template[0].format(**params)

class Conv2D(Layer):
    def initialize(self):
        if self.get_attr('data_format') == 'channels_last':
            shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        else:
            shape = [self.attributes['n_filt'], self.attributes['out_height'], self.attributes['out_width']]
            dims = ['N_FILT_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.add_weights(quantizer=self.get_attr('weight_quantizer'))
        self.add_bias(quantizer=self.get_attr('bias_quantizer'))
        if self.model.config.is_resource_strategy(self):
            self.set_attr('strategy', 'large')
            if self.model.config.backend.name == 'Vivado':
                self.model.config.backend.set_closest_reuse_factor(self)
                self.weights['weight'].data = np.transpose(self.weights['weight'].data, axes=[3, 2, 0, 1]) #(H,W,C,F) => (F,C,H,W)
        else:
            self.set_attr('strategy', 'latency')

    def function_cpp(self):
        params = self._default_function_params()
        params['strategy'] = self.get_attr('strategy')
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        params['w'] = self.get_weights('weight').name
        params['b'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        if self.get_attr('data_format') == 'channels_last':
            params['in_height'] = self.get_input_variable().dim_names[0]
            params['in_width'] = self.get_input_variable().dim_names[1]
            params['n_chan'] = self.get_input_variable().dim_names[2]
            params['out_height'] = self.get_output_variable().dim_names[0]
            params['out_width'] = self.get_output_variable().dim_names[1]
            params['n_filt'] = self.get_output_variable().dim_names[2]
        else:
            params['n_chan'] = self.get_input_variable().dim_names[0]
            params['in_height'] = self.get_input_variable().dim_names[1]
            params['in_width'] = self.get_input_variable().dim_names[2]
            params['n_filt'] = self.get_output_variable().dim_names[0]
            params['out_height'] = self.get_output_variable().dim_names[1]
            params['out_width'] = self.get_output_variable().dim_names[2]
        params['dilation'] = self.get_attr('dilation', 1)
        params['nzeros'] = self.get_weights('weight').nzeros
        params['config_t'] = 'std::nullptr_t'
        params['accum_t'] = self.get_precision_string(self.get_attr('accum_t'))

        if self.model.config.is_resource_strategy(self):
            params['config_t'] = 'config{}_mult'.format(self.index)
            conv_config = self._config_template[0].format(**params)

            mult_params = self._default_config_params()
            mult_params['n_in'] = self.get_attr('n_chan') * self.get_attr('filt_height') * self.get_attr('filt_width')
            mult_params['n_out'] = self.get_attr('n_filt')
            mult_config = self._config_template[1].format(**mult_params)

            return mult_config + '\n' + conv_config
        else:
            return self._config_template[0].format(**params)

class Pooling1D(Layer):
    def initialize(self):
        shape = [self.attributes['n_out'], self.attributes['n_filt']]
        dims = ['N_OUTPUTS_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

    def function_cpp(self):
        params = self._default_function_params()

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()
        params['n_out'] = self.get_output_variable().size_cpp()

        return self._config_template.format(**params)

class Pooling2D(Layer):
    def initialize(self):
        shape = [self.attributes['out_height'], self.attributes['out_width'], self.attributes['n_filt']]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_FILT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)
        self.set_attr('pool_op', self.get_attr('class_name').split('Pooling')[0])

    def function_cpp(self):
        params = self._default_function_params()
        params['data_format'] = 'cf' if self.get_attr('data_format') == 'channels_first' else 'cl'
        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().dim_names[0]
        params['in_width'] = self.get_input_variable().dim_names[1]
        params['out_height'] = self.get_output_variable().dim_names[0]
        params['out_width'] = self.get_output_variable().dim_names[1]
        params['n_filt'] = self.get_output_variable().dim_names[2]

        return self._config_template.format(**params)

class Activation(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)
        #if self.model.config.backend.name == 'Vivado' or self.model.config.backend.name == 'Quartus':
        if 'table_t' not in self.attributes:
            self.set_attr('table_t', FixedPrecisionType(width=18, integer=8))
        if 'table_size' not in self.attributes:
            self.set_attr('table_size', 1024)

    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self.get_attr('activation').lower()
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['type'] = self.get_attr('activation')
        params['n_in'] = self.get_input_variable().size_cpp()
        params['table_t'] = self.get_precision_string(self.get_attr('table_t'))

        return self._config_template.format(**params)



class ParametrizedActivation(Activation):
    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self._get_act_function_name()
        params['param'] = self.get_attr('activ_param', 1.0)
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

    def _get_act_function_name(self):
        act = self.get_attr('activation').lower()
        if act == 'leakyrelu':
            return 'leaky_relu'
        elif act == 'thresholdedrelu':
            return 'thresholded_relu'
        else:
            return act # ELU activation

class PReLU(Activation):
    def initialize(self):
        super(PReLU, self).initialize()
        self.add_weights_variable(name='alpha', var_name='a{index}')

    def function_cpp(self):
        params = self._default_function_params()
        params['activation'] = self.get_attr('activation').lower()
        params['param'] = self.get_weights('alpha').name
        params['config'] = '{}_config{}'.format(self.get_attr('activation'), self.index)

        return [self._function_template.format(**params)]

class Softmax(Activation):
    def initialize(self):
        super(Softmax, self).initialize()
        #if self.model.config.backend.name == 'Vivado' or self.model.config.backend.name == 'Quartus':
        if 'exp_table_t' not in self.attributes:
            self.set_attr('exp_table_t', self.get_attr('table_t'))
        if 'inv_table_t' not in self.attributes:
            self.set_attr('inv_table_t', self.get_attr('table_t'))

    def config_cpp(self):
        params = self._default_config_params()
        params['type'] = self.get_attr('activation')
        params['n_in'] = self.get_input_variable().size_cpp()
        params['exp_table_t'] = self.get_precision_string(self.get_attr('exp_table_t'))
        params['inv_table_t'] = self.get_precision_string(self.get_attr('inv_table_t'))

        return self._config_template.format(**params)

class BatchNormalization(Layer):
    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        self.add_output_variable(shape, dims)

        gamma = self.model.get_weights_data(self.name, 'gamma')
        beta = self.model.get_weights_data(self.name, 'beta')
        mean = self.model.get_weights_data(self.name, 'moving_mean')
        var = self.model.get_weights_data(self.name, 'moving_variance')

        scale = gamma / np.sqrt(var + self.get_attr('epsilon'))
        bias = beta - gamma * mean / np.sqrt(var + self.get_attr('epsilon'))
        print("hls_layers(784) - bias :", bias)
        self.add_weights_variable(name='scale', var_name='s{index}', data=scale)
        self.add_weights_variable(name='bias', var_name='b{index}', data=bias)

    def function_cpp(self):
        params = self._default_function_params()
        params['scale'] = self.get_weights('scale').name
        params['bias'] = self.get_weights('bias').name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_input_variable().size_cpp()

        return self._config_template.format(**params)

class Merge(Layer):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        shape = inp1.shape
        assert(inp1.shape == inp2.shape)
        dims = inp1.dim_names
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = {}
        params['merge'] = self.get_attr('op').lower()
        params['config'] = 'config{}'.format(self.index)
        params['input1_t'] = self.get_input_variable(self.inputs[0]).type.name
        params['input2_t'] = self.get_input_variable(self.inputs[1]).type.name
        params['output_t'] = self.get_output_variable().type.name
        params['input1'] = self.get_input_variable(self.inputs[0]).name
        params['input2'] = self.get_input_variable(self.inputs[1]).name
        params['output'] = self.get_output_variable().name

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_elem'] = self.get_input_variable(self.inputs[0]).size_cpp()

        return self._config_template.format(**params)

class Concatenate(Merge):
    def initialize(self):
        assert(len(self.inputs) == 2)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        axis = self.attributes['axis']
        shape = inp1.shape[:]
        shape[axis] += inp2.shape[axis]
        rank = len(shape)
        if rank > 1:
            dims = ['OUT_CONCAT_{}_{}'.format(i, self.index) for i in range(rank)]
        else:
            dims = ['OUT_CONCAT_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def config_cpp(self):
        params = self._default_config_params()
        for i in range(3):
            params.setdefault('n_elem1_{}'.format(i), 0)
            params.setdefault('n_elem2_{}'.format(i), 0)
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])
        for i, (s1, s2) in enumerate(zip(inp1.shape, inp2.shape)):
            params['n_elem1_{}'.format(i)] = s1
            params['n_elem2_{}'.format(i)] = s2

        return self._config_template.format(**params)

class BiasAdd(Merge): # TensorFlow's operator that gets merged into Dense/Conv
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        shape = inp.shape
        dims = inp.dim_names
        self.add_bias()
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        raise Exception('Layer {} should not be exported to HLS'.format(self.__class__.__name__))

    def config_cpp(self):
        raise Exception('Layer {} should not be exported to HLS'.format(self.__class__.__name__))

class Resize(Layer):
    def initialize(self):
        shape = [self.get_attr('new_height'), self.get_attr('new_width'), self.get_attr('n_chan')]
        dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index), 'N_CHAN_{}'.format(self.index)]
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['algorithm'] = self.get_attr('algorithm')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        return self._config_template.format(**params)

class Transpose(Layer):
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        perm = self.get_attr('perm')
        self.set_attr('dim', '{}d'.format(len(inp.shape)))
        if len(perm) == 4 and perm[0] == 0:
            perm = [i - 1 for i in perm[1:]]
        shape = [inp.shape[i] for i in perm]
        self.set_attr('perm_str', ','.join([str(i) for i in perm]))
        if len(shape) == 2:
            dims = ['OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', 1)
            self.set_attr('height', shape[0])
            self.set_attr('width', shape[1])
        else:
            dims = ['OUT_DEPTH_{}'.format(self.index), 'OUT_HEIGHT_{}'.format(self.index), 'OUT_WIDTH_{}'.format(self.index)]
            self.set_attr('depth', shape[0])
            self.set_attr('height', shape[1])
            self.set_attr('width', shape[2])
        self.add_output_variable(shape, dims)

    def function_cpp(self):
        params = self._default_function_params()
        params['dim'] = self.get_attr('dim')

        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()

        return self._config_template.format(**params)

class Lstm(Layer):

    def initialize(self):
        #Output data definitions
        shape = [self.get_attr('n_in')]
        print("hls_layers(911) - shape=10:", shape)
        dims = ['OUT_HEIGHT_{}'.format(self.index)]
        print("hls_layers(913) - dims:",dims)
        self.add_output_variable(shape, dims)

        data  = self.model.get_weights_data(self.name, 'kernel')
        data2 = self.model.get_weights_data(self.name, 'recurrent_kernel')
        data3 = self.model.get_weights_data(self.name, 'bias')
        #print("data3:")
        #print(data3)
        #print(self.get_layers.get_weights())
        weight_types=["i","f","c","o"]
        for i in range (0,4):
          self.add_weights_variable(name='weight_%s'% weight_types [i], var_name='kernel_%s_{index}' % weight_types [i], data=data[0][i*self.get_attr('n_in'):(i+1)*(self.get_attr('n_in'))], quantizer=self.get_attr('weight_quantizer'), compression=None)
          self.add_weights_variable(name='recurrent_weight_%s' % weight_types [i], var_name='recurrent_kernel_%s_{index}' % weight_types [i], data=data2[0:self.get_attr('n_in'),i*self.get_attr('n_in'):(i+1)*(self.get_attr('n_in'))], quantizer=self.get_attr('weight_quantizer'), compression=None)
          self.add_weights_variable(name='bias_%s'% weight_types [i], var_name='bias_%s_{index}' % weight_types [i], data=data3[i*self.get_attr('n_in'):(i+1)*(self.get_attr('n_in'))], quantizer=self.get_attr('weight_quantizer'), compression=None)


    def function_cpp(self):
        params = self._default_function_params()
         #if self.model.return_sequence:
        if self.model.sliding_window:
            params['input'] = params['input'] + '[0]'

        print("hls_layer(935) - params : ", params)
        print('hls_model(369) -get_output ', [i.shape for i in self.model.get_output_variables()])
        print("hls_layers(937) - lstm_input: ", params['input'])

        for i in self.model.get_output_variables():
            print("hls_teste", str(i))

        params['weights']=""
        for i in ["kernel","recurrent_kernel","bias"]:
          for j in ["i","f","c","o"]:
            params['weights'] += ""+ i + "_" + j + "_" + str(self.index)
            if not(i == "bias" and j == "o"):
              params['weights'] +=","
      #  params['algorithm'] = self.get_attr('algorithm')
        #print('Print 942',**params)
        print("hls_layer(943) - self._function_template.format(**params) :", self._function_template.format(**params))
        return [self._function_template.format(**params)]

    def config_cpp(self):
        params = self._default_config_params()
        params['n_in'] = self.get_attr('n_in')
        params['n_timestamp'] = self.get_attr('n_timestamp')

        return self._config_template.format(**params)


layer_map = {
    'Input'              : Input,
    'InputLayer'         : Input,
    'Activation'         : Activation,
    'QActivation'        : Activation,
    'LeakyReLU'          : ParametrizedActivation,
    'ThresholdedReLU'    : ParametrizedActivation,
    'ELU'                : ParametrizedActivation,
    'PReLU'              : PReLU,
    'Softmax'            : Softmax,
    'Reshape'            : Reshape,
    'Dense'              : Dense,
    'BinaryDense'        : Dense,
    'TernaryDense'       : Dense,
    'QDense'             : Dense,
    'Conv1D'             : Conv1D,
    'QConv1D'            : Conv1D,
    'Conv2D'             : Conv2D,
    'BinaryConv2D'       : Conv2D,
    'QConv2D'            : Conv2D,
    'BatchNormalization' : BatchNormalization,
    'MaxPooling1D'       : Pooling1D,
    'AveragePooling1D'   : Pooling1D,
    'MaxPooling2D'       : Pooling2D,
    'AveragePooling2D'   : Pooling2D,
    'Merge'              : Merge,
    'Concatenate'        : Concatenate,
    'Resize'             : Resize,
    'Transpose'          : Transpose,
    'LSTM'               : Lstm,
    # TensorFlow-specific layers:
    'BiasAdd'            : BiasAdd,
}

def register_layer(name, clazz):
    global layer_map
    layer_map[name] = clazz
