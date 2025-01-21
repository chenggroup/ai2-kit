import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import pickle
import os
import glob
from unicore.data import Dictionary

def get_result(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    predict = []
    target = []
    src_tokens = []
    for item in data:
        predict.extend(item['predict'].reshape(-1).tolist())
        target.extend(item['target'].reshape(-1).tolist())
        # print(item.keys())
        src_token = item["src_token"][item["select_atom"]==1]
        src_token = src_token.detach().cpu().numpy().tolist()
        src_tokens.extend(src_token)
    return target, predict, src_tokens

def reg_metrics(target, predict):
    r2 = r2_score(target, predict)
    mae = mean_absolute_error(target, predict)
    mse = mean_squared_error(target, predict)
    rmse = math.sqrt(mse)
    return r2, mae, mse, rmse

def plot_metrics(target, predict, save_path=None, element="All"):
    r2, mae, mse, rmse = reg_metrics(target, predict)
    plt.figure(figsize=(8, 8))
    plt.scatter(target, predict, color='blue', alpha=0.5)
    xy_max = max(max(target), max(predict))
    xy_min = min(min(target), min(predict))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='black', linestyle='--')
    plt.xlim(xy_min, xy_max)  
    plt.ylim(xy_min, xy_max)  
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted\nMAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}'.format(mae, rmse, r2))
    # plt.legend()
    if os.path.exists(save_path):
        fig_path = os.path.join(save_path, f'result_{element}.png')
        plt.savefig(fig_path)
    plt.show()


def main(args):
    dictionary = Dictionary.load(args.dict)
    if args.mode == 'cv':
        target = 0
        src_tokens = 0
        all_predict = []
        for folder in os.listdir(args.path):
            folder_path = os.path.join(args.path, folder)
            if os.path.isdir(folder_path):
                pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
                filename = pkl_files[0]
                target, predict, src_tokens = get_result(filename)
                all_predict.append(predict)
                plot_metrics(target, predict, folder_path)
                r2, mae, mse, rmse = reg_metrics(target, predict)
                print(f'metric of {filename}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

                elemenets = set(src_tokens)
                if len(elemenets) > 1:
                    for element in elemenets:
                        element_targets = np.array(target)[np.array(src_tokens)==element]
                        element_predicts = np.array(predict)[np.array(src_tokens)==element]
                        r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                        plot_metrics(target, predict, folder_path, dictionary[element])
                        print(f'metric of {dictionary[element]}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

        if all_predict:
            mean_predict = np.mean(np.vstack(all_predict), axis=0)
            plot_metrics(target, mean_predict, args.path)
            r2, mae, mse, rmse = reg_metrics(target, mean_predict)
            print(f'metric of mean\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
            elemenets = set(src_tokens)
            if len(elemenets) > 1:
                for element in elemenets:
                    element_targets = np.array(target)[np.array(src_tokens)==element]
                    element_predicts = np.array(mean_predict)[np.array(src_tokens)==element]
                    r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                    plot_metrics(element_targets, element_predicts, args.path, dictionary[element])
                    print(f'metric of {dictionary[element]}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

            mean_path = os.path.join(args.path, 'mean.pkl')
            mean_data = {'target': target, 'predict': mean_predict}
            with open(mean_path, 'wb') as file:
                pickle.dump(mean_data, file)

    else :
        pkl_files = glob.glob(os.path.join(args.path, "*.pkl"))
        filename = pkl_files[0]
        target, predict, src_tokens = get_result(filename)
        r2, mae, mse, rmse = reg_metrics(target, predict)
        print(f'metric of {filename}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
        plot_metrics(target, predict, args.path)
        elemenets = set(src_tokens)
        if len(elemenets) > 1:
            for element in elemenets:
                element_targets = np.array(target)[np.array(src_tokens)==element]
                element_predicts = np.array(predict)[np.array(src_tokens)==element]
                r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                plot_metrics(target, predict, args.path, dictionary[element])
                print(f'metric of {dictionary[element]}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--path', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--dict', type=str)
    args = parser.parse_args()
    main(args)