import torch
import numpy as np
from robomimic.scripts.train import train

from arguments import get_args
from model_old import FSQAE
from util import multiplyList
from dataset import MotionDataset
from dataset import get_data_loaders


class TokenDecoder():

    def __init__(self):
        # 2, load model
        self.model = FSQAE()
        self.model.cuda(torch.cuda.current_device())
        # original saved file with DataParallel
        ckpt_path = '/home/milleret/FSQ-pytorch-main/checkpoints/2025-10-10 11:11:11_fsq-n_embed-1000_g1__add_ee_loss/ckpts/best_model.pt'
        state_dict = torch.load(ckpt_path)['model_state_dict']

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def ids_to_tokens(self, ids):

        return self.model.quantize.indices_to_codes(ids)

    def ar_token_decode(self, model, prev_codes, cur_code):
        '''
        model: FSQ model
        prev_codes: (B, seq_len, num_levels), torch.Tensor
        cur_code: (B, new_seq_len, num_levels), torch.Tensor
        '''

        codes = torch.cat((prev_codes, cur_code), dim=1)

        reconstructions = model.decode(codes)
        return reconstructions


def main():
    args = get_args()

    assert args.quantizer == 'fsq'


    npz_path = '/home/milleret/whole_body_tracking/data/tokenize/g1_20fps_0918.npz'
    train_set = MotionDataset(npz_path, split='train')
    val_set = MotionDataset(npz_path, split='val')
    val_set = train_set

    # 1, load dataset
    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False
    )


    # 2, load model
    model = FSQAE()
    model.cuda(torch.cuda.current_device())
    # original saved file with DataParallel
    ckpt_path = '/home/milleret/FSQ-pytorch-main/checkpoints/2025-10-10 11:11:11_fsq-n_embed-1000_g1__add_ee_loss/ckpts/best_model.pt'
    state_dict = torch.load(ckpt_path)['model_state_dict']

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()


    get_l1_loss = torch.nn.L1Loss()

    # for compute codebook usage
    num_embed = multiplyList([8, 8, 8, 6, 5])
    codebook_usage = dict()

    total_l1_loss = 0
    num_iter = 0

    data_dict_all = []

    input_motion, root_pos, root_rot = train_set.get_normalized_motion_clip(0, 345668)
    input_motion = torch.from_numpy(input_motion).unsqueeze(0).to(dtype=torch.float32)

    token_decoder = TokenDecoder()

    with torch.no_grad():
        input_motion = input_motion.cuda(torch.cuda.current_device())

        # codes = model.get_code(input_motion)
        # codes = codes.cpu().numpy()
        # np.save('test/codes.npy', codes)
        # breakpoint()
        codes = np.load('test/codes.npy')  # (1, 86417, 5)
        codes = torch.from_numpy(codes).cuda(torch.cuda.current_device())

        reconstructions_all = []

        clip_len = 9   # code len
        overlap_len = 8

        prev_codes = codes[:, :overlap_len, :]

        for i in range(overlap_len, 7000, clip_len - overlap_len):

            cur_code = codes[:, [i], :]

            reconstructions = token_decoder.ar_token_decode(model, prev_codes, cur_code)
            prev_codes = torch.cat((prev_codes[:, 1:, :], cur_code), dim=1)

            # code = codes[:, i:i+clip_len, :]  # (1, clip_len, 5)
            # reconstructions = model.decode(code)

            if len(reconstructions_all) == 0:
                reconstructions_all.append(reconstructions[:, :, :])
            else:
                reconstructions_all.append(reconstructions[:, (overlap_len)*4:, :])

        reconstructions_all = torch.cat(reconstructions_all, dim=1)
        reconstructions = reconstructions_all
        # reconstructions, ids = model(input_motion, return_id=True)

        # ids = torch.flatten(ids)
        # for quan_id in ids:
        #     codebook_usage[quan_id.item()] = codebook_usage.get(quan_id.item(), 0) + 1

    # l1loss = get_l1_loss(input_motion, reconstructions)
    # total_l1_loss += l1loss.cpu().item()

    raw_data = train_set.denormalize_torch(input_motion).cpu().numpy()
    rec_data = train_set.denormalize_torch(reconstructions).cpu().numpy()


    data_dict = {
        'raw_data': raw_data,
        'rec_data': rec_data,
        'raw_root_pos': root_pos,
        'raw_root_rot': root_rot
    }
    data_dict_all.append(data_dict)

    # save
    np.save('test/sample_data.npy', data_dict_all, allow_pickle=True)


if __name__ == "__main__":
    main()