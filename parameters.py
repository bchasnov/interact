import numpy as np
from constants import *
import benpy as bp

import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree

class InteractParameter(pTypes.GroupParameter):
    def __init__(self, params, changing, changed, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.values = {}
        self.callback_changing = changing
        self.callback_changed = changed
        self.initialize(params)

    def initialize(self, params):
        for name, value in params:
            self.values[name] = value
            if type(value) is float: 
                self.addChild(dict(name=name, value=value, type='float', step=.1))
            if type(value) is int: 
                self.addChild(dict(name=name, value=value, type='int'))
            if type(value) is bool: 
                self.addChild(dict(name=name, value=value, type='bool'))
            p = self.param(name)
            p.sigValueChanged.connect(self.changed)
            p.sigValueChanging.connect(self.changing)

    def changed(self):
        for name in self.values:
            self.values[name] = self.param(name).value()
        self.callback_changed(self.values)

    def changing(self, param, value):
        self.values[param.name()] = value;
        self.callback_changing(self.values)
        
