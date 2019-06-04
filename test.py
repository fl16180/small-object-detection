import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config, resume):
    logger = config.get_logger('test')

    data_loader = config.initialize('data_loader', module_data, mode='test')

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for batch_idx, (data, boxes, labels, _) in enumerate(test_data_loader):

            data = data.to(DEVICE)
            boxes = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            output_boxes, output_scores = self.model(data)

            mbloss = loss_fn(threshold=0.5, neg_pos_ratio=3,
                                     alpha=1., device=device)
            loss = mbloss(output_boxes, output_scores, boxes, labels)

            # computing loss, metrics on test set
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    config = ConfigParser(args)
    main(config, args.resume)
