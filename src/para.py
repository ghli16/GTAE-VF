def parser_add_main_args(parser):
    # dataset, protocol
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--knn_nums', type=int, default=25, help='number of k for KNN graph')

    parser.add_argument('--kfolds', type=int, default=5)



    # encoder hyper-parameter for
    parser.add_argument('--num_node_features', default=1280, type=int, help='initial feature dimension in GCN.')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--method', type=str, default='difformer')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.2, help='weight for residual link')
    parser.add_argument('--use_bn', type=bool, default=True, help='use layernorm')
    parser.add_argument('--use_residual', type=bool, default=True, help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', type=bool, default=True, help='use pos emb')
    parser.add_argument('--use_weight', type=bool, default=False, help='use weight for GNN convolution')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])
    # encoder hyper-parameter for
    parser.add_argument('--num_node_features1', default=256, type=int, help='initial feature dimension in GCN.')
    parser.add_argument('--hidden_channels1', type=int, default=1280)
    parser.add_argument('--out_channels1', type=int, default=1)
    parser.add_argument('--method1', type=str, default='difformer')
    parser.add_argument('--num_layers1', type=int, default=2, help='number of layers for deep methods')
    parser.add_argument('--num_heads1', type=int, default=4, help='number of heads for attention')
    parser.add_argument('--alpha1', type=float, default=0.2, help='weight for residual link')
    parser.add_argument('--use_bn1', type=bool, default=True, help='use layernorm')
    parser.add_argument('--use_residual1', type=bool, default=True, help='use residual link for each GNN layer')
    parser.add_argument('--use_graph1', type=bool, default=True, help='use pos emb')
    parser.add_argument('--use_weight1', type=bool, default=False, help='use weight for GNN convolution')
    parser.add_argument('--kernel1', type=str, default='simple', choices=['simple', 'sigmoid'])
    # training
    parser.add_argument('--mlpin_dim', type=float, default=256, help='mlpini_dim.')
    parser.add_argument('--mlp1_dim', type=float, default=1, help='mlpmid_dim.')
    parser.add_argument('--mlp2_dim', type=float, default=64, help='mlpini_dim.')
    parser.add_argument('--mlp3_dim', type=float, default=1, help='mlpini_dim.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10000, help='mini batch training for large graphs')
