'''
### notes:
Create a library/class that starting with dataframes, lists, ... generates the json  
this library should identify all relevant parameters required by the json  
create a function to save the json and/or execute mlprocess
Should be integrated on mlproces and aligned


'''
import joblib
import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatType , FloatTensorType
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import load_model
import onnx
import tf2onnx
from onnx import helper, TensorProto
import numpy as np

NODE_NAMES= ["Applicability", "Preprocess", "ML_Model", "PostProcess", "Uncertainty"]

class ModelTracker():  #OrderedDict??
    def __init__(self):
        self.nodes = []
        self.Applicability = None
        self.Preprocess = None
      
    def add_node(self, node_name):
        if node_name == "Applicability":
            self.Applicability = {"Meta":{},
                              "Global" : {},
                             }
            self.nodes.append("Applicability")
        elif node_name == "Preprocess":
            self.Preprocess = {"Inputs":[],
                           "Transformers":{},
                           "Outputs": [],
                           "Association":{}
                          }
            self.nodes.append("Preprocess")
        elif node_name == "ML_Model":
            self.ML_Model = {"Inputs":[],
                           "Models":{},
                           "Outputs": [],
                           "Association":{}
                          }
            self.nodes.append("ML_Model")

        elif node_name == "PostProcess":
            
            self.PostProcess = {"Inputs":[],
                           "Transformers":{},
                           "Outputs": [],
                           "Association":{}
                          }
            self.nodes.append("PostProcess")
        elif node_name == "Uncertainty":
            self.Uncertainty = {"Local":{},
                            "Global" : {},
                          }
        else:
            print(f"node name {node_name} not valid")

    def drop_node(self, node_name):
        if node_name in self.nodes:
            self[node_name] = None
            self.nodes.pop(node_name)

        else:
            print(f"node name {node_name} not valid")
            

    def update_Applicability(self, Inputs, Outputs ):
        """
        :data: Input_dataframe
        
        :return: dict of applicability
        """
        self.Applicability["Global"] = {"Inputs": Inputs.describe().loc[["min", "max"]].to_dict(),
                                        "Outputs": Outputs.describe().loc[["min", "max"]].to_dict()
                                       }
        
        return self.Applicability["Global"]
    
    def update_Preprocess(self, inputs, transformer_name, transformer, outputs):
        #salvar el fichero transformer.
        file = f"{transformer_name}.pkl"
        if inputs != None:
            print("change model inputs", inputs)
            inputs_fix = [input.lower() for input in inputs]
            transformer.feature_names_in_ = inputs_fix
        joblib.dump(transformer, file) 

        i = len(self.Preprocess["Transformers"])
        print(inputs)
        print(self.Preprocess["Inputs"])
        self.Preprocess["Inputs"] = list(dict.fromkeys(self.Preprocess["Inputs"]+ inputs))
        self.Preprocess["Transformers"].update({i:{'type':str(transformer.__class__).split("'")[1].split(".")[0],
                                                   'file':file
                                                   }
                                               })
        self.Preprocess["Outputs"] = list(dict.fromkeys(self.Preprocess["Outputs"]+ outputs))
        self.Preprocess["Association"].update({i:{"Inputs": inputs,
                                                            "Outputs": outputs
                                                            }
                                              })
        
    def update_Preprocessonnx(self, inputs, transformer_name, transformer, outputs):
        
        transformertype= str(transformer.__class__).split("'")[1].split(".")[0]
        if transformertype=="str":
            i = len(self.Preprocess["Transformers"])
            print(inputs)
            print(self.Preprocess["Inputs"])
            self.Preprocess["Inputs"] = list(dict.fromkeys(self.Preprocess["Inputs"]+ inputs))
            self.Preprocess["Transformers"].update({i:{'type':"pandas",
                                                    'formula': transformer,
                                                    }
                                                })
            self.Preprocess["Outputs"] = list(dict.fromkeys(self.Preprocess["Outputs"]+ outputs))
            self.Preprocess["Association"].update({i:{"Inputs": inputs,
                                                                "Outputs": outputs
                                                                }
                                                })
            
            
        else:
            
            #salvar el fichero transformer.
            file_onnx = f"{transformer_name}.onnx"
            print(inputs)
            #print(outputs)
            initial_type = [("inputs", FloatTensorType([None,len(inputs)]))]
            onnx_pre = convert_sklearn(transformer,initial_types=initial_type, target_opset={'': 17, 'ai.onnx.ml': 2})
            with open(file_onnx,"wb") as f:
                f.write(onnx_pre.SerializeToString())

            i = len(self.Preprocess["Transformers"])
            print(inputs)
            print(self.Preprocess["Inputs"])
            self.Preprocess["Inputs"] = list(dict.fromkeys(self.Preprocess["Inputs"]+ inputs))
            self.Preprocess["Transformers"].update({i:{'type':str(onnx_pre.__class__).split("'")[1].split(".")[0],
                                                    'file':file_onnx
                                                    }
                                                })
            self.Preprocess["Outputs"] = list(dict.fromkeys(self.Preprocess["Outputs"]+ outputs))
            self.Preprocess["Association"].update({i:{"Inputs": inputs,
                                                                "Outputs": outputs
                                                                }
                                                })
        
    def update_ML_Model(self, inputs, model_name, model, outputs, time_series_parameters=None):
        print(model_name)
        file = model_name# f"{model_name}.h5"
        print("saving model", file)
        modeltype= str(model.__class__).split("'")[1].split(".")[0]
        if modeltype == 'sklearn':
            if inputs != None:
                print("change model inputs", inputs)
                model.feature_names_in_ = inputs
            joblib.dump(model, file) 
        else:
            model.save(f"{file}")
        
        #if len(self.ML_Model["Models"]) == 0:
        i = len(self.ML_Model["Models"])
        #else:
        #    i = 0
        print(len(self.ML_Model["Models"]))
        
        self.ML_Model["Inputs"] = list(dict.fromkeys(self.ML_Model["Inputs"] + inputs))
        if time_series_parameters is None:
            if modeltype == 'sklearn':
                self.ML_Model["Models"].update({i:{'type':modeltype,
                                                'file':str(file),
                                                # 'batch_size' : 4096,
                                                }
                                        })
            else:
                self.ML_Model["Models"].update({i:{'type':modeltype,
                                                'file':str(file),
                                                'batch_size' : 4096,
                                                }
                                        })
        else:
            self.ML_Model["Models"].update({i:{'type':modeltype,
                                              'file':str(file),
                                              "time_series_parameters":time_series_parameters,
                                             }
                                       })
                                  
        '''self.ML_Model["Models"].update({'type':str(model.__class__).split("'")[1].split(".")[0],
                                              'file':str(file)
                                             })'''
        self.ML_Model["Outputs"] = list(dict.fromkeys(self.ML_Model["Outputs"] + outputs))
        self.ML_Model["Association"].update({i:{"Inputs": inputs,
                                                            "Outputs": outputs
                                                }
                                            })
    
    def update_ML_Modelonnx(self, inputs, model_name, model, outputs, time_series_parameters=None):
        print(model_name)
        file = f"{model_name}.onnx"
        print("saving model", file)
        modeltype= str(model.__class__).split("'")[1].split(".")[0]
        if modeltype == 'sklearn':
            initial_type = [("inputs", FloatTensorType([None,len(inputs)]))]
            onnx_pre = convert_sklearn(model,initial_types=initial_type, target_opset={'': 17, 'ai.onnx.ml': 2})
            with open(file,"wb") as f:
                f.write(onnx_pre.SerializeToString())
        else:
            model.save(f"{file}")
        
        #if len(self.ML_Model["Models"]) == 0:
        i = len(self.ML_Model["Models"])
        #else:
        #    i = 0
        print(len(self.ML_Model["Models"]))
        
        self.ML_Model["Inputs"] = list(dict.fromkeys(self.ML_Model["Inputs"] + inputs))
        if time_series_parameters is None:
            self.ML_Model["Models"].update({i:{'type':str(onnx_pre.__class__).split("'")[1].split(".")[0],
                                              'file':str(file)
                                             }
                                       })
        else:
            self.ML_Model["Models"].update({i:{'type':modeltype,
                                              'file':str(file),
                                              "time_series_parameters":time_series_parameters,
                                             }
                                       })
                                  
        self.ML_Model["Outputs"] = list(dict.fromkeys(self.ML_Model["Outputs"] + outputs))
        self.ML_Model["Association"].update({i:{"Inputs": inputs,
                                                            "Outputs": outputs
                                                }
                                            })
        
    def update_ML_resXnetModelonnx(self, inputs, model_name, model, outputs, time_series_parameters=None):
        print(model_name)
        file = f"{model_name}.onnx"
        print("saving model", file)
        modeltype= str(model.__class__).split("'")[1].split(".")[0]
        if modeltype == 'keras':
            onnx_model, _ = tf2onnx.convert.from_keras(model, opset = 17)
            onnx.save(onnx_model, file)
        else:
            model.save(f"{file}")
        
        #if len(self.ML_Model["Models"]) == 0:
        i = len(self.ML_Model["Models"])
        #else:
        #    i = 0
        print(len(self.ML_Model["Models"]))
        
        self.ML_Model["Inputs"] = list(dict.fromkeys(self.ML_Model["Inputs"] + inputs))
        if time_series_parameters is None:
            self.ML_Model["Models"].update({i:{'type':str(onnx_model.__class__).split("'")[1].split(".")[0],
                                              'file':str(file),
                                              'batch_size':4096
                                             }
                                       })
        else:
            self.ML_Model["Models"].update({i:{'type':modeltype,
                                              'file':str(file),
                                              "time_series_parameters":time_series_parameters,
                                             }
                                       })
                                  
        '''self.ML_Model["Models"].update({'type':str(model.__class__).split("'")[1].split(".")[0],
                                              'file':str(file)
                                             })'''
        self.ML_Model["Outputs"] = list(dict.fromkeys(self.ML_Model["Outputs"] + outputs))
        self.ML_Model["Association"].update({i:{"Inputs": inputs,
                                                            "Outputs": outputs
                                                }
                                            })



    def build_postprocessor(input_name, output_name):

            Clip = helper.make_node('Clip', inputs=['lamb_inv_value', 'min_v', 'max_v'], outputs=[output_name], name='Clip152')
            Inv = helper.make_node("Div", inputs = ["unit", input_name], outputs=["inv_value"], name = "Inv")
            Mul = helper.make_node("Mul", inputs=["lamb", "inv_value"], outputs=["lamb_inv_value"])


            graph_def = helper.make_graph(
            nodes = [Inv, Clip, Mul],
            name = 'test_graph',
            inputs = [helper.make_tensor_value_info('unit', TensorProto.FLOAT, []), helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, None]),
                        helper.make_tensor_value_info('min_v', TensorProto.FLOAT, []), helper.make_tensor_value_info('max_v', TensorProto.FLOAT, []),
                        helper.make_tensor_value_info('lamb', TensorProto.FLOAT, [None, None]), #helper.make_tensor_value_info('clip_output', TensorProto.FLOAT, [None]),
                    ], # use your input
            outputs = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, None])], # use your output
            initializer = [helper.make_tensor('unit', TensorProto.FLOAT, [], [1.]), helper.make_tensor('min_v', TensorProto.FLOAT, [], [0.]), helper.make_tensor('max_v', TensorProto.FLOAT, [], [100.])])
            
            model = onnx.helper.make_model_gen_version(graph_def)
            with open(f"postprocess_{input_name}_min_lambda_rf_inv_100.onnx", "wb") as f:
                f.write(model.SerializeToString())
                
            return model

    def update_PostProcess_onnx(self, inputs, transformer_name, transformer, outputs):
        ## TODO

        transformertype= str(transformer.__class__).split("'")[1].split(".")[0]
        if transformertype=="str":
                i = len(self.PostProcess["Transformers"])
                print(inputs)
                print(self.Preprocess["Inputs"])
                self.PostProcess["Inputs"] = list(dict.fromkeys(self.PostProcess["Inputs"]+ inputs))
                self.PostProcess["Transformers"].update({i:{'type':"pandas",
                                                        'formula': transformer,
                                                        }
                                                    })
                self.PostProcess["Outputs"] = list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs))
                self.PostProcess["Association"].update({i:{"Inputs": inputs,
                                                                    "Outputs": outputs
                                                                    }
                                                    })
        else:
            # if transformer_name!=None:
            file_onnx = f"{transformer_name}.onnx"
            onnx_post = build_postprocessor(inputs, outputs)
            with open(file_onnx,"wb") as f:
                f.write(onnx_post.SerializeToString())
            print(inputs)
            print(self.PostProcess["Inputs"])

            self.PostProcess["Inputs"] = list(dict.fromkeys(self.PostProcess["Inputs"]+ inputs))
            self.PostProcess["Transformers"].update({i:{'type':str(transformer.__class__).split("'")[1].split(".")[0],
                                                    'file':file_onnx
                                                    }
                                                })
            self.PostProcess["Outputs"] = list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs))
            self.PostProcess["Association"].update({i:{"Inputs": inputs,
                                                                "Outputs": outputs
                                                                }
                                                })

            # else:

            #     self.PostProcess["Inputs"] = list(dict.fromkeys(self.PostProcess["Inputs"] + inputs))
            #     self.PostProcess["Transformers"] = {}
            #     self.PostProcess["Outputs"] = list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs))
            #     self.PostProcess["Association"] = {"Inputs":list(dict.fromkeys(self.PostProcess["Inputs"] + inputs)),
            #                                     "Outputs":  list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs)),  
            #                                     }

    def update_PostProcess(self, inputs, transformer_name, transformer, outputs):
        ## TODO

        transformertype= str(transformer.__class__).split("'")[1].split(".")[0]
        if transformertype=="str":
                i = len(self.PostProcess["Transformers"])
                print(inputs)
                print(self.Preprocess["Inputs"])
                self.PostProcess["Inputs"] = list(dict.fromkeys(self.PostProcess["Inputs"]+ inputs))
                self.PostProcess["Transformers"].update({i:{'type':"pandas",
                                                        'formula': transformer,
                                                        }
                                                    })
                self.PostProcess["Outputs"] = list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs))
                self.PostProcess["Association"].update({i:{"Inputs": inputs,
                                                                    "Outputs": outputs
                                                                    }
                                                    })

        else:
            if transformer_name!=None:
                file = f"{transformer_name}.pkl"
                joblib.dump(transformer, file) 
                
                i = len(self.PostProcess["Transformers"])
                print(inputs)
                print(self.Preprocess["Inputs"])
                self.PostProcess["Inputs"] = list(dict.fromkeys(self.PostProcess["Inputs"]+ inputs))
                self.PostProcess["Transformers"].update({i:{'type':str(transformer.__class__).split("'")[1].split(".")[0],
                                                        'file':file
                                                        }
                                                    })
                self.PostProcess["Outputs"] = list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs))
                self.PostProcess["Association"].update({i:{"Inputs": inputs,
                                                                    "Outputs": outputs
                                                                    }
                                                    })

            else:

                self.PostProcess["Inputs"] = list(dict.fromkeys(self.PostProcess["Inputs"] + inputs))
                self.PostProcess["Transformers"] = {}
                self.PostProcess["Outputs"] = list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs))
                self.PostProcess["Association"] = {"Inputs":list(dict.fromkeys(self.PostProcess["Inputs"] + inputs)),
                                                "Outputs":  list(dict.fromkeys(self.PostProcess["Outputs"]+ outputs)),  
                                                }
                
    def to_json(self):
        output = {}
        for node in self.nodes:
            if node == "Applicability":
                output[node] = self.Applicability
            elif node == "Preprocess":
                output[node] = self.Preprocess
            elif node == "ML_Model":
                output["ML Model"] = self.ML_Model
            elif node == "PostProcess":
                output["PostProcess"] = self.PostProcess
        return json.dumps(output, indent = 4)
    