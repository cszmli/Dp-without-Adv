import json
import logging
import math
import os
import zipfile
from typing import Dict
import copy
import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, Field
from allennlp.data.instance import Instance
from overrides import overrides

import sys
sys.path.append('/home/zli1/ConvLab/')
from convlab.lib.file_util import cached_path
from convlab.modules.dst.multiwoz.rule_dst import RuleDST
from convlab.modules.policy.system.multiwoz.util import ActionVocab
from convlab.modules.state_encoder.multiwoz.multiwoz_state_encoder import MultiWozStateEncoder
from convlab.modules.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# @DatasetReader.register("mle_policy")
class MlePolicyDatasetReader(DatasetReader):
    """
    Reads instances from a data file:

    Parameters
    ----------
    """
    def __init__(self,
                 num_actions: int,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.dst = RuleDST()
        self.action_vocab = ActionVocab(num_actions=num_actions)
        self.action_list = self.action_vocab.vocab
        one_c, two_c, other_c = 0,0,0
        dm_c1, dm_c2 = 0, 0
        for x in self.action_list[:300]:
            if len(x.keys())==1:
                one_c +=1 
            elif len(x.keys())==2:
                two_c += 1
            else:
                other_c += 1
            print(x)
        print(one_c, two_c, other_c, one_c+two_c+other_c)
        # raise ValueError('stop')
        self.build_action_info()
        # self.exclude_domain = ['hotel','Hotel']
        self.exclude_domain = []


        self.state_encoder = MultiWozStateEncoder()
        
        self.keep_index = list(range(0,30)) + list(range(297, 303)) + list(range(308, 320)) \
                     + list(range(322, 325)) + list(range(330, 351)) + list(range(353, 365)) \
                     + list(range(369, 384)) + list(range(387, 392))
        self.extend_index = []
        for i in range(392):
            if i not in self.keep_index:
                self.extend_index.append(i)
                
    def _int2binary_9(self,x):
        return list(reversed( [(x >> i) & 1 for i in range(9)]))

    @overrides
    def _read(self, file_path):
        data = []
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        if file_path.endswith("zip"):
            archive = zipfile.ZipFile(file_path, "r")
            data_file = archive.open(os.path.basename(file_path)[:-4])
        else:
            data_file = open(file_path, "r")

        logger.info("Reading instances from lines in file at: %s", file_path)

        dialogs = json.load(data_file)

        for dial_name in dialogs:
            dialog = dialogs[dial_name]["log"]
            self.dst.init_session()
            data_sub = []
            for i, turn in enumerate(dialog):
                if i % 2 == 0:  # user turn
                    self.dst.update(user_act=turn["dialog_act"])
                else:  # system turn
                    delex_act = {}
                    for domain_act in turn["dialog_act"]:
                        domain, act_type = domain_act.split('-', 1)
                        if domain in self.exclude_domain: #
                            continue   #
                        if act_type in ['NoOffer', 'OfferBook']:
                            delex_act[domain_act] = ['none']
                        elif act_type in ['Select']:
                            for sv in turn["dialog_act"][domain_act]:
                                if sv[0] != "none":
                                    delex_act[domain_act] = [sv[0]]
                                    break
                        else:
                            delex_act[domain_act] = [sv[0] for sv in turn["dialog_act"][domain_act]]
                    state_vector = self.state_encoder.encode(self.dst.state)   
                    if len(delex_act)==0:
                        continue         
                    action_index = self.find_best_delex_act(delex_act)
                    action_rep_delax = self.action_rep_list[action_index]
                    action_index_binary = self._int2binary_9(action_index)
                    s = state_vector.ravel().tolist()
                    s_one_hot = self.convert_to_onehot(s)
                    data_sub.append({'state_onehot':s_one_hot, 'state_convlab':s, \
                                 'action':action_index, 'action_binary': action_index_binary,
                                 'action_rep_seg': action_rep_delax})
                    # data.append({'state_onehot':s_one_hot, 'state_convlab':s, \
                    #              'action':action_index, 'action_binary': action_index_binary,
                    #              'action_rep_seg': action_rep_delax})
            data_sub_final = []
            for i,turn in enumerate(data_sub):
                if i < len(data_sub)-1:
                    next_state = data_sub[i+1]['state_convlab']
                else:
                    next_state = turn['state_convlab']
                turn['state_convlab_next'] = next_state
                data_sub_final.append(turn)
            data += data_sub_final
        new_path = file_path.replace('json.zip', 'sa_alldomain_withnext.json')
        action_rep_path =  './data/multiwoz/action_rep_seg.json'
        with open(new_path, 'w', encoding="utf8") as f:
            json.dump(data, f, indent=2)
        # with open(action_rep_path, 'w', encoding="utf8") as f:
        #     json.dump(self.action_rep_list, f, indent=2)
                    # yield self.text_to_instance(state_vector, action_index)

    def find_best_delex_act(self, action):
        def _score(a1, a2):
            score = 0
            for domain_act in a1:
                if domain_act not in a2:
                    score += len(a1[domain_act])
                else:
                    score += len(set(a1[domain_act]) - set(a2[domain_act]))
            return score

        best_p_action_index = -1
        best_p_score = math.inf
        best_pn_action_index = -1
        best_pn_score = math.inf
        for i, v_action in enumerate(self.action_list):
            if v_action == action:
                return i
            else:
                p_score = _score(action, v_action)
                n_score = _score(v_action, action)
                if p_score > 0 and n_score == 0 and p_score < best_p_score:
                    best_p_action_index = i
                    best_p_score = p_score
                else:
                    if p_score + n_score < best_pn_score:
                        best_pn_action_index = i
                        best_pn_score = p_score + n_score
        if best_p_action_index >= 0:
            return best_p_action_index
        return best_pn_action_index

    def text_to_instance(self, state: np.ndarray, action: int = None) -> Instance:  # type: ignore
        """
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields["states"] = ArrayField(state)
        if action is not None:
            fields["actions"] = LabelField(action, skip_indexing=True)
        return Instance(fields)
    
    def convert_to_onehot(self, state):
        
        new_v = []
        for i, v in enumerate(state):
            if i in self.keep_index:
                new_v.append(v)
            elif i in self.extend_index:
                hd = [0,0]
                hd[int(v)]=1.0
                new_v += hd 
        return new_v

    def build_action_info(self):
        #TODO: project the action ID into a vector which contains domain, Intent, and slot
        domain_dict, act_dict, slot_dict = {}, {}, {}
        c1, c2, c3=0,0,0
        self.action_rep_list = []
        domain_counter, act_counter, slot_counter = 0, 0, 0
        act_placeholder = [0] * 10 + [0] * 14 + [1, 0] * 28
        act_placeholder[9], act_placeholder[23] = 1, 1
        for sys_action in self.action_list:
            action_reps = []
            for domain_type in sys_action.keys():
                action_rep = [0] * 10 + [0] * 14 + [1, 0] * 28
                domain, tpe = domain_type.lower().split('-')

                if domain not in domain_dict:     #0
                    domain_dict[domain]=domain_counter
                    domain_counter += 1
                action_rep[domain_dict[domain]] = 1

                if tpe not in act_dict:       #10
                    act_dict[tpe] = act_counter
                    act_counter += 1
                action_rep[10 + act_dict[tpe]] = 1

                for slot in sys_action[domain_type]:  # 24
                    if slot not in slot_dict:
                        slot_dict[slot] = slot_counter
                        slot_counter += 1
                    action_rep[24 + 2 * slot_dict[slot]] = 0
                    action_rep[24 + 2 * slot_dict[slot]+1] = 1
                action_reps = action_reps + action_rep
            if len(action_reps)==80:
                action_reps = action_reps + act_placeholder
                c1+=1
            elif len(action_reps)==240:
                action_reps = action_reps[:160]
                c3+=1
            self.action_rep_list.append(copy.deepcopy(action_reps))
        assert len(self.action_rep_list)==300       
        # print(domain_counter, act_counter, slot_counter)
        


            
            
    
if __name__ == '__main__':
    datareader = MlePolicyDatasetReader(300)
    train_path = './data/multiwoz/train.json.zip'
    test_path = './data/multiwoz/test.json.zip'
    val_path = './data/multiwoz/val.json.zip'
    datareader._read(test_path)
    datareader._read(val_path)
    datareader._read(train_path)
    print("Data Extracting Done")
    
    
    

