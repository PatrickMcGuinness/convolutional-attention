import numpy as np
import sys

from convolutional_attention.copy_conv_rec_learner import ConvolutionalCopyAttentionalRecurrentLearner
fname = sys.argv[1]
model2 = ConvolutionalCopyAttentionalRecurrentLearner.load(fname)

test_file = sys.argv[2]
test_data, original_names = model2.naming_data.get_data_in_recurrent_copy_convolution_format(test_file, model2.padding_size)
test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data


idx = 10
print 'ORIGINAL',' '.join(original_names[idx].split(','))
print 'CODE', ' '.join(test_code[idx])

import cPickle
with open(fname, 'rb') as f:
    learner = cPickle.load(f)
model2.model.restore_parameters(learner.parameters)

res = model2.predict_name(np.atleast_2d(test_code[idx]))

for r,v in res:
    print v, ' '.join(r)