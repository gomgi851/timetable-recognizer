import json,unittest
from pathlib import Path
import torch
from room_model import CHARACTERS,decode
from data import edit_distance,score,image_tensor

class TrainingTests(unittest.TestCase):
    def test_ctc_repeats_and_blank_are_decoded_without_dictionary(self):
        chars=['R','R',None,'8',None,'8',None,'7']
        values=torch.full((1,len(chars),len(CHARACTERS)+1),-10.)
        for t,c in enumerate(chars):values[0,t,0 if c is None else CHARACTERS.index(c)+1]=10
        self.assertEqual(decode(values),['R887'])
    def test_missing_prediction_counts_as_error(self):
        self.assertEqual(score([{'label':'B102'}],[''])['accuracy'],0)
        self.assertEqual(edit_distance('B102','8102'),1)
    def test_missing_roi_has_valid_white_input(self):
        image=image_tensor({'roi':None})
        self.assertEqual(tuple(image.shape),(1,64,320))
        self.assertTrue(torch.all(image==1))
    def test_source_partitions_do_not_overlap(self):
        p=json.loads(Path('training/protocol.json').read_text(encoding='utf-8'))
        train,val,test=[set(p[key])for key in ['trainImages','validationImages','testImages']]
        self.assertFalse(train&val or train&test or val&test)

if __name__=='__main__':unittest.main()
