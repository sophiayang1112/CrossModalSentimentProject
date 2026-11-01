import io
import unittest
import numpy as np
import torch
from crossmodal_sentiment.data import simulate,prepare,partitions,Vocabulary,collate
from crossmodal_sentiment.network import SentimentNetwork,MFB
from crossmodal_sentiment.cli import evaluate


class DataTests(unittest.TestCase):
    def test_future_opinions_excluded(self):
        raw=simulate(60,7)[0];prepared=prepare(raw)
        for stream in ["recency","popularity"]:
            self.assertFalse(any("future" in row["opinion_id"] for row in prepared[stream]))

    def test_future_engagement_not_used(self):
        raw=simulate(60,7)[0]
        for row in raw["popularity"]:row["engagement_observed_at"]="2099-01-01"
        with self.assertRaises(ValueError):prepare(raw)

    def test_company_groups_disjoint(self):
        samples=[prepare(row) for row in simulate(60,7)]
        parts=partitions(samples)
        groups=[set(samples[i]["company_id"] for i in indices) for indices in parts.values()]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(sum(map(len,parts.values())),len(samples))

    def test_vocabulary_unknown_token(self):
        sample=prepare(simulate(60,7)[0]);vocab=Vocabulary().fit([sample])
        self.assertEqual(vocab.encode("heldoutuniquetoken",4),[1])
        self.assertEqual(vocab.mapping["[PAD]"],0)

    def test_weighted_recall_equals_accuracy(self):
        labels=np.array([0,1,2,1]);p=np.array([[.8,.1,.1],[.6,.2,.2],[.1,.1,.8],[.1,.8,.1]])
        metrics=evaluate(labels,p)
        self.assertAlmostEqual(metrics["accuracy"],metrics["weighted_recall"])


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def inputs(self):
        return torch.tensor([[[2,3,0],[4,0,0]],[[2,4,5],[0,0,0]]]),torch.tensor([[[5,6,0]],[[3,2,0]]])

    def model(self,variant="full"):
        torch.manual_seed(5)
        return SentimentNetwork(12,dim=16,heads=4,factors=2,max_opinions=5,dropout=0,variant=variant)

    def test_shape_and_attention_normalization(self):
        model=self.model().eval();a,b=self.inputs()
        logits,weights=model(a,b,True)
        self.assertEqual(tuple(logits.shape),(2,3))
        self.assertEqual(tuple(weights["trending_to_timely"].shape),(2,4,2,3))
        torch.testing.assert_close(weights["trending_to_timely"].sum(-1),torch.ones((2,4,2)))
        torch.testing.assert_close(weights["trending_to_timely"][1,:,:,2],torch.zeros((4,2)))

    def test_padding_invariance(self):
        model=self.model().eval();a,b=self.inputs()
        extended=torch.cat([a,torch.zeros((2,1,3),dtype=torch.long)],dim=1)
        torch.testing.assert_close(model(a,b),model(extended,b),atol=1e-6,rtol=1e-5)

    def test_batch_matches_individual(self):
        model=self.model().eval();a,b=self.inputs()
        together=model(a,b)
        for i in range(2):torch.testing.assert_close(together[i],model(a[i:i+1],b[i:i+1])[0],atol=1e-6,rtol=1e-5)

    def test_gradients_reach_cross_attention_and_fusion(self):
        model=self.model();a,b=self.inputs()
        loss=torch.nn.functional.cross_entropy(model(a,b),torch.tensor([0,2]));loss.backward()
        for parameter in [model.stage_one.in_proj_weight,model.stage_two.in_proj_weight,model.mfb.left.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0)

    def test_mfb_factor_sum(self):
        layer=MFB(2,2,2)
        with torch.no_grad():
            layer.left.weight.fill_(1);layer.right.weight.fill_(1)
        result=layer(torch.ones((1,2)),torch.ones((1,2)))
        torch.testing.assert_close(result,torch.ones((1,2))/np.sqrt(2))

    def test_single_stream_independence(self):
        a,b=self.inputs();changed=b.clone();changed[changed>0]=8
        model=self.model("recency_only").eval()
        torch.testing.assert_close(model(a,b),model(a,changed))

    def test_state_dict_round_trip(self):
        model=self.model().eval();a,b=self.inputs()
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        restored=self.model().eval();restored.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(model(a,b),restored(a,b))


if __name__=="__main__":unittest.main()
