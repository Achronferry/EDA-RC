import yaml

def constract_models(args, in_size, save_hyper_param=None):
    modules = args.model_type.split('_')
    use_num_predict = modules[-1].lower() == 'np'
    if modules[0] == 'EEND':
        from models.EEND import EEND
        model = EEND(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                num_predict=use_num_predict
                )
        if save_hyper_param is not None:
            with open(save_hyper_param, mode='w') as cf:
                yaml.dump({'num_speakers': args.num_speakers,
                    'in_size': in_size,
                    'hidden_size': args.hidden_size,
                    'transformer_encoder_n_heads': args.transformer_encoder_n_heads,
                    'transformer_encoder_n_layers': args.transformer_encoder_n_layers,
                    'transformer_encoder_dropout': args.transformer_encoder_dropout,
                    } ,cf)
            
    elif modules[0] == 'RPEEEND':
        from models.tranformer_rp import TransformerLinearModel_RP
        model = TransformerLinearModel_RP(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                max_relative_position=args.max_relative_position,
                gap=args.gap,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                num_predict=use_num_predict
                )
        if save_hyper_param is not None:
            with open(args.model_save_dir + "/param.yaml", mode='w') as cf:
                yaml.dump({'num_speakers': args.num_speakers,
                    'in_size': in_size,
                    'hidden_size': args.hidden_size,
                    'transformer_encoder_n_heads': args.transformer_encoder_n_heads,
                    'transformer_encoder_n_layers': args.transformer_encoder_n_layers,
                    'max_relative_position': args.max_relative_position,
                    'gap': args.gap,
                    'transformer_encoder_dropout': args.transformer_encoder_dropout,
                    } ,cf)

    elif modules[0] == 'EENDC':
        from models.EENDC import EENDC
        model = EENDC(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                num_predict=use_num_predict
                )
        if save_hyper_param is not None:
            with open(args.model_save_dir + "/param.yaml", mode='w') as cf:
                yaml.dump({'num_speakers': args.num_speakers,
                    'in_size': in_size,
                    'hidden_size': args.hidden_size,
                    'transformer_encoder_n_heads': args.transformer_encoder_n_heads,
                    'transformer_encoder_n_layers': args.transformer_encoder_n_layers,
                    'transformer_encoder_dropout': args.transformer_encoder_dropout,
                    } ,cf)
    else:
        raise ValueError('Possible model_type is "Transformer"')

    return model