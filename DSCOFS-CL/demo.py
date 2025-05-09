import torch
from fun.network import Networks
from fun import contrastive_loss
import numpy as np
import warnings
import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.io import savemat
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score


mat_files = ["Isolet"]
# lamdas = [0.5]
# percentS = [0.1,0.2, 0.3, 0.4, 0.5]
# alphas = [0.000001, 0.0001, 0.01, 1, 100, 10000, 1000000]
# betas =  [0.000001, 0.0001, 0.01, 1, 100, 10000, 1000000]
# gammas = [0.000001, 0.0001, 0.01, 1, 100, 10000, 1000000]
# mus =    [0.000001, 0.0001, 0.01, 1, 100, 10000, 1000000]

lamdas = [0.5]
percentS = [0.3]
alphas = [0.0001]
betas =  [1]
gammas = [10000]
mus =    [100]


for datasheet in mat_files:
    loss_alls = []
    others_results = []
    ACC_10_tO_100s = []
    NMI_10_tO_100s = []
    for lamda in lamdas:
        for percent in percentS:
            for alpha in alphas:
                for beta in betas:
                    for gamma in gammas:
                        for mu in mus:
                            seed = 42
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                            torch.backends.cudnn.deterministic = True
                            torch.backends.cudnn.benchmark = False

                            mat_path = "data/" + datasheet
                            data_0 = sio.loadmat(mat_path)
                            A0 = data_0['X']
                            labels = data_0['Y'] - 1
                            A = A0 - A0.mean(axis=0)
                            train_num = A.shape[0]
                            dim = A.shape[1]
                            unique_numbers = np.unique(labels)
                            class_num = len(unique_numbers)
                            temp = np.array([0 for l in range(len(labels))])
                            for k in range(len(labels)):
                                temp[k] = labels[k][0]
                            train_data = A.astype(np.float32)
                            train_data = torch.tensor(train_data)

                            warnings.filterwarnings("ignore")

                            instance_temperature_1 = 0.5
                            instance_temperature_2 = 0.5
                            loss_device = torch.device("cuda")
                            instance_1 = contrastive_loss.InstanceLoss(train_num, instance_temperature_1,
                                                                                     loss_device).to(loss_device)
                            instance_2 = contrastive_loss.InstanceLoss(train_num, instance_temperature_2,
                                                                                     loss_device).to(loss_device)

                            model = Networks(class_num, train_num, dim).cuda()
                            lr1 = 0.001
                            optimizer_X = torch.optim.Adam([model.X], lr=lr1, weight_decay=1e-4)
                            optimizer_M = torch.optim.Adam([model.M], lr=lr1, weight_decay=1e-4)
                            n_epochs = 500
                            loss_all = []
                            Q = torch.zeros(train_num, train_num).cuda()
                            T = torch.zeros(dim, class_num).cuda()
                            P = torch.zeros(dim, class_num).cuda()

                            for epoch in range(n_epochs + 1):
                                input = train_data.view(train_num, dim).cuda()
                                output1, output2, X, Z = model(input)
                                prjspace = input.mm(X)

                                loss_contrastive_1 = instance_1(input, output1)
                                loss_contrastive_2 = instance_2(prjspace, output2)
                                loss_X = torch.norm(X.T.mm(X) - torch.eye(class_num).cuda(), p=2).pow(2)

                                Q_pre = Q.clone().detach()
                                if torch.isnan(Z).any() or torch.isinf(Z).any():
                                    print("Z 包含 NaN/Inf，修正中...")
                                    Z = torch.nan_to_num(Z)
                                    break
                                Q_update = (Z + 0.001 * Q_pre) / 1.001

                                try:
                                    U, sigma, Vt = torch.linalg.svd(Q_update)
                                    rk = int(0.1 * len(sigma))
                                    U_k = U[:, :rk]
                                    sigma_k = torch.diag(sigma[:rk])
                                    Vt_k = Vt[:rk, :]
                                    Q = U_k @ sigma_k @ Vt_k
                                except torch._C._LinAlgError:
                                    print("SVD 失败，尝试ing")
                                    if torch.isnan(Q_update).any() or torch.isinf(Q_update).any():
                                        print("Q_update 包含 NaN 或 Inf，进行修正")
                                        Q_update = torch.nan_to_num(Q_update)
                                        try:
                                            U, sigma, Vt = torch.linalg.svd(Q_update)
                                            rk = int(0.1 * len(sigma))
                                            U_k = U[:, :rk]
                                            sigma_k = torch.diag(sigma[:rk])
                                            Vt_k = Vt[:rk, :]
                                            Q = U_k @ sigma_k @ Vt_k
                                        except torch._C._LinAlgError:
                                            print("SVD 失败，保持 Q 不变")
                                            Q = Q.clone().detach().requires_grad_(True)

                                T_pre = T.clone().detach()
                                T_update = (X + 0.001 * T_pre) / 1.001
                                norm_W = torch.diag(T_update.mm(T_update.T)).pow(0.5)
                                _, index = torch.sort(norm_W, descending=True)
                                top_100 = index[0:100]
                                T = torch.zeros_like(X)
                                T[top_100] = T_update[top_100]

                                P_pre = P.clone().detach()
                                P_update = (X + 0.001 * P_pre) / 1.001
                                P_update_flattened = P_update.flatten()
                                P_update_abs_flattened = torch.abs(P_update_flattened)
                                _, top_p_indices_P = torch.topk(P_update_abs_flattened, int(dim * class_num * percent))
                                P = torch.zeros_like(X)
                                P.view(-1)[top_p_indices_P] = P_update_flattened[top_p_indices_P]

                                loss_new1 = torch.norm(Z - Q, p='fro').pow(2)
                                loss_new2 = torch.norm(X - T, p='fro').pow(2)
                                loss_new3 = torch.norm(X - P, p='fro').pow(2)

                                loss = loss = (lamda * loss_contrastive_1 + (1 - lamda) * loss_contrastive_2
                                        + alpha * loss_new1 + beta * loss_new2 + gamma * loss_new3 + mu * loss_X)
                                loss_all.append(loss.item())
                                
                                if epoch % 50 == 0:
                                    print(
                                        "Epoch {}/{}, Loss is:{:.4f}, lamda:{}, percent:{}, alpha:{}, beta:{}, gamma:{}, mu:{}".format(
                                            epoch, n_epochs, loss.item(), lamda, percent, alpha, beta, gamma, mu
                                        ),
                                        end="\r",
                                    )
                                    print()
                                    print("--------------------------------")
                                    norm_W = torch.diag(X.mm(X.T)).pow(0.5)
                                    _, index = torch.sort(norm_W, descending=True)
                                    top_100 = index[0:100]
                                    top_100 = top_100.cpu().numpy()
                                    file_name = f"result/{datasheet}_{lamda}_{percent}_{alpha}_{beta}_{gamma}_{mu}_top100.mat"
                                    savemat(file_name, {'top100': top_100})
                                    print(f"top100: {top_100}")
                                    

                                loss1 = ( (1 - lamda) * loss_contrastive_2 + beta * loss_new2 + gamma * loss_new3 + mu * loss_X)
                                optimizer_X.zero_grad()
                                loss1.backward()
                                optimizer_X.step()

                                output1, output2, X, Z = model(input)
                                prjspace = input.mm(X)
                                loss_contrastive_1 = instance_1(input, output1)
                                loss_contrastive_2 = instance_2(prjspace, output2)
                                loss_new1 = torch.norm(Z - Q, p='fro').pow(2)
                                
                                loss2 = (lamda * loss_contrastive_1 + (1 - lamda) * loss_contrastive_2 + alpha * loss_new1 )                     
                                optimizer_M.zero_grad()
                                loss2.backward() 
                                optimizer_M.step()


