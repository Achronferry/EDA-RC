import yaml
import torch

def constract_models(args, in_size, save_hyper_param=None):
    modules = args.model_type.split('+')
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

    elif modules[0] == 'EEND_EDA':
        from models.EEND_EDA import EEND_EDA
        model = EEND_EDA(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                num_predict=use_num_predict
                )
        if args.inherit_from is not None:
            param_dict = torch.load(args.inherit_from)
            def del_prefix(module_keys):
                split_key = module_keys.split('.')
                split_key = split_key if split_key[0] != 'module' else split_key[1:]
                return '.'.join(split_key)
            filtered_params = {del_prefix(i):param_dict[i]  for i in list(param_dict.keys()) if 'encoder' in i}
            miss, unexpect = model.load_state_dict(filtered_params, strict=False)
            assert unexpect == [] , f'Miss key(s): {miss} \nUnexpect key(s):{unexpect}'
            
        if save_hyper_param is not None:
            with open(save_hyper_param, mode='w') as cf:
                yaml.dump({
                    'in_size': in_size,
                    'hidden_size': args.hidden_size,
                    'transformer_encoder_n_heads': args.transformer_encoder_n_heads,
                    'transformer_encoder_n_layers': args.transformer_encoder_n_layers,
                    'transformer_encoder_dropout': args.transformer_encoder_dropout,
                    } ,cf)    

    
    elif modules[0] == 'EDA_RC':
        from models.EDA_RC import EDA_RC
        model = EDA_RC(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                )
        if args.inherit_from is not None:
            param_dict = torch.load(args.inherit_from)
            def del_prefix(module_keys):
                split_key = module_keys.split('.')
                split_key = split_key if split_key[0] != 'module' else split_key[1:]
                return '.'.join(split_key)
            filtered_params = {del_prefix(i):param_dict[i]  for i in list(param_dict.keys()) if 'encoder' in i or 'decoder' in i}
            miss, unexpect = model.load_state_dict(filtered_params, strict=False)
            for n, p in model.named_parameters():
                if n in filtered_params:
                    p.require_grad=False
            assert unexpect == [] , f'Miss key(s): {miss} \nUnexpect key(s):{unexpect}'
            
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
                yaml.dump({
                    'in_size': in_size,
                    'hidden_size': args.hidden_size,
                    'transformer_encoder_n_heads': args.transformer_encoder_n_heads,
                    'transformer_encoder_n_layers': args.transformer_encoder_n_layers,
                    'transformer_encoder_dropout': args.transformer_encoder_dropout,
                    } ,cf)
    elif modules[0] == 'EEND_GRID':
        from models.EEND_GRID import EEND_GRID
        model = EEND_GRID(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                )
        if args.inherit_from is not None:
            param_dict = torch.load(args.inherit_from)
            def del_prefix(module_keys):
                split_key = module_keys.split('.')
                split_key = split_key if split_key[0] != 'module' else split_key[1:]
                return '.'.join(split_key)
            filtered_params = {del_prefix(i):param_dict[i]  for i in list(param_dict.keys()) if 'encoder' in i}
            miss, unexpect = model.load_state_dict(filtered_params, strict=False)
            assert unexpect == [] , f'Miss key(s): {miss} \nUnexpect key(s):{unexpect}'
        
        if save_hyper_param is not None:
            with open(args.model_save_dir + "/param.yaml", mode='w') as cf:
                yaml.dump({
                    'in_size': in_size,
                    'hidden_size': args.hidden_size,
                    'transformer_encoder_n_heads': args.transformer_encoder_n_heads,
                    'transformer_encoder_n_layers': args.transformer_encoder_n_layers,
                    'transformer_encoder_dropout': args.transformer_encoder_dropout,
                    } ,cf)
    elif modules[0] == 'SC_EEND':
        from models.SC_EEND import SC_EEND
        model = SC_EEND(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                )
        if args.inherit_from is not None:
            param_dict = torch.load(args.inherit_from)
            def del_prefix(module_keys):
                split_key = module_keys.split('.')
                split_key = split_key if split_key[0] != 'module' else split_key[1:]
                return '.'.join(split_key)
            filtered_params = {del_prefix(i):param_dict[i]  for i in list(param_dict.keys()) if 'encoder' in i}
            miss, unexpect = model.load_state_dict(filtered_params, strict=False)
            assert unexpect == [] , f'Miss key(s): {miss} \nUnexpect key(s):{unexpect}'
            
        if save_hyper_param is not None:
            with open(args.model_save_dir + "/param.yaml", mode='w') as cf:
                yaml.dump({
                    'in_size': in_size,
                    'hidden_size': args.hidden_size,
                    'transformer_encoder_n_heads': args.transformer_encoder_n_heads,
                    'transformer_encoder_n_layers': args.transformer_encoder_n_layers,
                    'transformer_encoder_dropout': args.transformer_encoder_dropout,
                    } ,cf)
    else:
        raise ValueError('Possible model_type is "Transformer"')

    return model