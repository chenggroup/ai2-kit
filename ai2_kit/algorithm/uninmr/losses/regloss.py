# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from unicore.losses import UnicoreLoss, register_loss

ATTR_REGESTRY = {
    ### for MAP task
    ### log1p + standardization
    'hmof': [0.7352541603735866, 0.6419512242050728, 'log1p_standardization'],
    'CoRE_DB': [4.877768551522245, 1.3630433513018894, 'log1p_standardization'],
    'CoRE_MAP': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    'CoRE_MAP_CH4': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    'CoRE_MAP_CO2': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    'CoRE_MAP_Ar': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    'CoRE_MAP_Kr': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    'CoRE_MAP_Xe': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    'CoRE_MAP_O2': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    'CoRE_MAP_N2': [1.318703908155812, 1.657051374039756,'log1p_standardization'],

    'CoRE_MAP_LargeScale': [1.318703908155812, 1.657051374039756,'log1p_standardization'],
    
    'EXP_ADS': [1.318703908155812, 1.657051374039756, 'log1p_standardization'],
    'EXP_ADS_hmof': [0.7352541603735866, 0.6419512242050728, 'log1p_standardization'],

    ### single task for CoRE_O2N2
    ### log + standardization
    # 'HenrysconstantN2':[-12.554020498803377, 0.8169898691823003, 'log_standardization'], 
    # 'HenrysconstantO2':[-12.584561891924537, 0.7245019267012439, 'log_standardization'],
    # 'N2uptakemolkg':[-1.1498171780249011, 0.675196328280587, 'log_standardization'],
    # 'O2uptakemolkg':[-1.1475385472764816, 0.658851890288826, 'log_standardization'],
    # 'SelfdiffusionofN2cm2s':[-9.053232476305368, 0.9922367202982066, 'log_standardization'],
    # 'SelfdiffusionofO2cm2s':[-8.961047510953662, 0.8288661462404647, 'log_standardization'],
    # 'SelfdiffusionofN2cm2sinfDilute':[-8.947461648760754, 0.9206194145843283, 'log_standardization'],
    # 'SelfdiffusionofO2cm2sinfDilute':[-8.884236846328324, 0.7933940931623032, 'log_standardization'],

    ### rescale + log + standardization
    'HenrysconstantN2':[-1.0410950338331513, 0.8169898691823003, 'log_standardization'], 
    'HenrysconstantO2':[-1.071636426954307, 0.7245019267012439, 'log_standardization'],
    'N2uptakemolkg':[0.23647718309498947, 0.675196328280587, 'log_standardization'],
    'O2uptakemolkg':[0.23875581384340877, 0.658851890288826, 'log_standardization'],
    'SelfdiffusionofN2cm2s':[0.15710789567081626, 0.9922367202982066, 'log_standardization'],
    'SelfdiffusionofO2cm2s':[0.2492928610225201, 0.8288661462404647, 'log_standardization'],
    'SelfdiffusionofN2cm2sinfDilute':[0.26287872321543004, 0.9206194145843283, 'log_standardization'],
    'SelfdiffusionofO2cm2sinfDilute':[0.32610352564785905, 0.7933940931623032, 'log_standardization'],

    ### single task for hmof 
    #### standardization
    'CarbonDioxide_298_0.01': [0.10396354700540493, 0.22647223711266543, 'standardization'],
    'CarbonDioxide_298_0.05': [0.3552362138732747, 0.4999100688274694, 'standardization'],
    'CarbonDioxide_298_0.1': [0.5899521612050606, 0.6954814707840283, 'standardization'],
    'CarbonDioxide_298_0.5': [1.7881394809316975, 1.4182319573028714, 'standardization'],
    'CarbonDioxide_298_2.5': [4.893662963716865, 2.7665924837196814, 'standardization'],
    'Krypton_273_1.0': [0.8142831247196857, 0.49019273224067283, 'standardization'],
    'Krypton_273_10.0': [3.457146423970651, 1.844647949833803, 'standardization'],
    'Krypton_273_5.0': [2.3621157288824643, 1.1836266090761243, 'standardization'],
    'Methane_298_0.05': [0.06919052531491733, 0.11732131578468759, 'standardization'],
    'Methane_298_0.5': [0.5079206304786128, 0.48712780952084545, 'standardization'],
    'Methane_298_0.9': [0.810676617550032, 0.6673173034373471, 'standardization'],
    'Methane_298_2.5': [1.7398582577817177, 1.1031611422278471, 'standardization'],
    'Methane_298_4.5': [2.6132505457714963, 1.4531130345320122, 'standardization'],
    'Nitrogen_298_0.09': [0.030780563865632177, 0.02929089458934474, 'standardization'],
    'Nitrogen_298_0.9': [0.2797187385989161, 0.189400059544328, 'standardization'],
    'Xenon_273_1.0': [1.4689066891996336, 1.3300085891403677, 'standardization'],
    'Xenon_273_10.0': [3.9755578904359625, 2.4003920253199396, 'standardization'],
    'Xenon_273_5.0': [3.1119706247761525, 2.0242093202015337, 'standardization'],

    ### single task for CoRE_2019_N2-Ar
    ### log1p + standardization
    'N2_77_1334.0': [5.005959612727335, 0.9380200664108198, 'log1p_standardization'],
    'N2_77_4.0': [4.496992971461004, 1.1038735067142493, 'log1p_standardization'],
    'N2_77_316.0': [4.954693194444789, 0.9423899959164476, 'log1p_standardization'],
    'N2_77_23714.0': [5.040552599629131, 0.9313133637633051, 'log1p_standardization'],
    'N2_77_100000.0': [5.043832659155757, 0.9249850622789892, 'log1p_standardization'],
    'N2_77_5623.0': [5.030385999255922, 0.9362231694882166, 'log1p_standardization'],
    'N2_77_75.0': [4.856944591940231, 0.9566910917561435, 'log1p_standardization'],
    'N2_77_18.0': [4.713678987977164, 1.0070195140317848, 'log1p_standardization'],
    'N2_77_1.0': [4.230620623412422, 1.234397064939272, 'log1p_standardization'],
    'Ar_87_0.001': [1.264925291061282, 1.4867134690024557, 'log1p_standardization'],
    'Ar_87_0.01': [2.066426365172186, 1.666100149041618, 'log1p_standardization'],
    'Ar_87_0.1': [2.9871127303390046, 1.658046175780731, 'log1p_standardization'],
    'Ar_87_1.0': [3.816935011472259, 1.4580307128906818, 'log1p_standardization'],
    'Ar_87_10.0': [4.45813211359122, 1.1737996300082165, 'log1p_standardization'],
    'Ar_87_100.0': [4.905501842880746, 0.9790022884273142, 'log1p_standardization'],
    'Ar_87_1000.0': [5.1573076668377835, 0.92729736281114, 'log1p_standardization'],
    'Ar_87_2000.0': [5.202870468197515, 0.9249116774555552, 'log1p_standardization'],
    'Ar_87_4000.0': [5.241115877609449, 0.9270222084856665, 'log1p_standardization'],
    'Ar_87_6000.0': [5.259917500374311, 0.929448501473627, 'log1p_standardization'],
    'Ar_87_8000.0': [5.271761191168934, 0.9312142306996114, 'log1p_standardization'],
    'Ar_87_10000.0': [5.280589108694357, 0.9321831016872058, 'log1p_standardization'],
    'Ar_87_12000.0': [5.286852623781105, 0.9328796920711562, 'log1p_standardization'],
    'Ar_87_14000.0': [5.292341463307119, 0.9338449322238297, 'log1p_standardization'],
    'Ar_87_16000.0': [5.2972633994853835, 0.9346733153719622, 'log1p_standardization'],
    'Ar_87_18000.0': [5.3013194272697275, 0.935470730977557, 'log1p_standardization'],
    'Ar_87_20000.0': [5.3047892965617995, 0.9359557206685724, 'log1p_standardization'],
    'Ar_87_24000.0': [5.309577127060796, 0.9363774096872732, 'log1p_standardization'],
    'Ar_87_28000.0': [5.314111952981034, 0.9365075712907492, 'log1p_standardization'],
    'Ar_87_30000.0': [5.316011853742724, 0.9367891977656196, 'log1p_standardization'],
    'Ar_87_32000.0': [5.3179287846496885, 0.9366727374288051, 'log1p_standardization'],
    'Ar_87_36000.0': [5.320997819154503, 0.9366032897930457, 'log1p_standardization'],
    'Ar_87_40000.0': [5.323674724298496, 0.9368878628042351, 'log1p_standardization'],
    'Ar_87_50000.0': [5.328890590191506, 0.9368744610042196, 'log1p_standardization'],
    'Ar_87_60000.0': [5.333144071131759, 0.9369344127309929, 'log1p_standardization'],
    'Ar_87_70000.0': [5.336719947125746, 0.9366400471272878, 'log1p_standardization'],
    'Ar_87_80000.0': [5.33972128498756, 0.9365754297888167, 'log1p_standardization'],
    'Ar_87_90000.0': [5.340562553912887, 0.9372271176303055, 'log1p_standardization'],
    'Ar_87_100000.0': [5.340956577497255, 0.9371543533069057, 'log1p_standardization'],
    'N2_77_500.0': [5.037110113212742, 0.8900966851333135, 'log1p_standardization'],
    'N2_77_80.0': [4.912906318894842, 0.9148506244526978, 'log1p_standardization'],
    'N2_77_1000.0': [5.065140379743582, 0.8863932618423059, 'log1p_standardization'],
    'N2_77_200.0': [4.986376743515708, 0.9009131838615344, 'log1p_standardization'],
    'N2_77_5000.0': [5.089621749968469, 0.8889060941021609, 'log1p_standardization'],
    'N2_77_1500.0': [5.075729564131175, 0.8847394236449164, 'log1p_standardization'],
    'N2_77_10.0': [4.657726187443544, 1.0093572655049672, 'log1p_standardization'],
    'N2_77_2000.0': [5.073938058650238, 0.884034678108839, 'log1p_standardization'],
    'N2_77_20000.0': [5.106830900365691, 0.8988000188416835, 'log1p_standardization'],
    'N2_77_40000.0': [5.106670847473756, 0.8980085525936815, 'log1p_standardization'],
    'N2_77_60000.0': [5.100358262950352, 0.8914529601571912, 'log1p_standardization'],
    'N2_77_99900.0': [5.09161251433794, 0.885958368636817, 'log1p_standardization'],
    'N2_77_80000.0': [5.09540545828945, 0.8886990881959167, 'log1p_standardization'],

    ### single task for CoRE_structure_feature
    ### log1p + standardization
    # 'CoRE_PLD': [1.650712, 0.36442, 'standardization'],
    # 'CoRE_LCD': [1.963333, 0.382678, 'standardization'],
    # 'CoRE_volume': [0.386544, 0.215881, 'standardization'],    
    # 'CoRE_VF': [0.422403, 0.080336, 'standardization'],
    'hMOF_PLD': [1.865408, 0.497615, 'log1p_standardization'],
    'hMOF_LCD': [2.157661, 0.463281, 'log1p_standardization'],
    'hMOF_volume': [0.648911, 0.341842, 'log1p_standardization'],
    'hMOF_VF': [0.497774, 0.09921, 'log1p_standardization'],
    'CoRE_PLD': [4.63215803544683, 2.834787,'standardization'],
    'CoRE_LCD': [6.751438881677484, 3.905345916111568, 'standardization'],
    'CoRE_VF':[0.5305805441837245, 0.12409350367710244, 'standardization'],
    'CoRE_volume':[0.5205515104842735,0.8092927979609419, 'standardization'],


    ### single task
    ### log1p + standardization
    'Henry_C12_100K': [11.941814829776213, 5.745713382152701, 'log1p_standardization'],
    'D_C12': [1.898171e-08, 1.904578e-08, 'log1p_standardization'],
}


class Normalization(object):
    def __init__(self, mean=None, std=None, normal_type=None):
        self.mean = mean
        self.std = std
        self.normal_type = normal_type
    
    def transform(self, x):
        if self.normal_type == 'log1p_standardization':
            return (torch.log1p(x) - self.mean) / self.std
        elif self.normal_type == 'standardization':
            return (x - self.mean) / self.std
        else:
            raise ValueError('normal_type should be log1p_standardization or standardization')
    
    def inverse_transform(self, x):
        if self.normal_type == 'log1p_standardization':
            return torch.expm1(x * self.std + self.mean)
        elif self.normal_type == 'standardization':
            return x * self.std + self.mean
        else:
            raise ValueError('normal_type should be log1p_standardization or standardization')

@register_loss("mof_v1_mse")   
class MOFV1MSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], 
                           features_only=True, 
                           classification_head_name=self.args.classification_head_name)
        sample_size = sample["target"]["finetune_target"].size(0)
        _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
        normalizer = Normalization(_mean, _std, _normal_type)
        loss = self.compute_loss(model, net_output[0], sample, normalizer, reduce=reduce)
        if not self.training:
            logging_output = {
                "loss": loss.data,
                "predict": normalizer.inverse_transform(net_output[0].view(-1, self.args.num_classes).data),
                "target": sample["target"]["finetune_target"].view(-1, self.args.num_classes).data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
            }   
        logging_output['bsz'] = sample_size          
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, normalizer, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = sample['target']['finetune_target'].view(-1, self.args.num_classes).float()
        normalize_targets = normalizer.transform(targets)
        loss = F.mse_loss(
            predicts,
            normalize_targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        num_tasks = logging_outputs[0]["num_task"]
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / num_tasks, sample_size, round=3
        )
        if 'valid' in split or 'test' in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            predicts = predicts.detach().cpu().numpy()
            targets = torch.cat([log.get("target") for log in logging_outputs], dim=0)
            targets = targets.detach().cpu().numpy()
            ##### 
            r2 = r2_score(targets, predicts)
            metrics.log_scalar("{}_r2".format(split), r2, sample_size, round=3)
            

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("mof_v2_mse")
class MOFV2MSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], 
                           features_only=True, 
                           classification_head_name=self.args.classification_head_name)
        _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
        normalizer = Normalization(_mean, _std, _normal_type)
        loss = self.compute_loss(model, net_output[0], sample, normalizer, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            logging_output = {
                "loss": loss.data,
                "predict": normalizer.inverse_transform(net_output[0].view(-1, self.args.num_classes).data),
                "target": sample["target"]["finetune_target"].view(-1, self.args.num_classes).data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "task_name": sample["task_name"],
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
            }   
        logging_output['bsz'] = sample_size          
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, normalizer, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = sample['target']['finetune_target'].view(-1, self.args.num_classes).float()
        normalize_targets = normalizer.transform(targets)
        loss = F.mse_loss(
            predicts,
            normalize_targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        num_tasks = logging_outputs[0]["num_task"]
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / num_tasks, sample_size, round=3
        )
        if 'valid' in split or 'test' in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            predicts = predicts.detach().cpu().numpy()
            targets = torch.cat([log.get("target") for log in logging_outputs], dim=0)
            targets = targets.detach().cpu().numpy()
            task_name_list = [item for log in logging_outputs for item in log.get("task_name")]
            predicts = predicts.reshape(-1,)
            targets = targets.reshape(-1,)
            ##### 
            r2 = r2_score(targets, predicts)
            metrics.log_scalar("{}_r2".format(split), r2, sample_size, round=4)

            if 'test' in split:
                df = pd.DataFrame({'predicts': predicts, 'targets': targets, 'task_name': task_name_list})
                if df['task_name'].nunique() < 20 and df['task_name'].nunique() > 1:
                    stats = df.groupby('task_name').apply(lambda x: r2_score(x['targets'], x['predicts'])).to_dict()
                    for task_name, r2 in stats.items():
                        metrics.log_scalar("{}_r2_{}".format(split, task_name), r2, sample_size, round=4)
            

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
