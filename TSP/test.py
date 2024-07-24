# from GNN.data import GoogleTSPReader
# import torch
# from GNN.solvers.ctsp import get_convtsp_model
# from GNN.gnn_utils import *
#
# val = "GNN/ConvTSP/TSP20/tsp20_val_concorde.txt"
# dataset = GoogleTSPReader(100, -1, 4, val)
# problems = torch.rand(size=(4, 100, 2))
# batch = dataset.process_batch(problems)
#
# base_model, _ = get_convtsp_model(0)
# base_model.cuda()
# params = torch.load("GNN/trained_ctsp.pt")
# params_new = dict()
# for key in params.keys():
#     params_new[key[7:]] = params[key]
# base_model.load_state_dict(params_new)
#
# aa, loss = base_model(batch)
# pred_tour_len, gt_tour_len = padded_get_stats(aa, batch)

