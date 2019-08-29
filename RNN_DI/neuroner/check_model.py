from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow


chkp.print_tensors_in_checkpoint_file("./trained_models/i2b2_2014_glove_spacy_bioes/model.ckpt",tensor_name=None, all_tensors=False)

reader = pywrap_tensorflow.NewCheckpointReader("./trained_models/i2b2_2014_glove_spacy_bioes/model.ckpt")
output_bias = reader.get_tensor('token_lstm/bidirectional_LSTM/forward/initial_cell_state')
