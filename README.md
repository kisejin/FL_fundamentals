# I. Introduction:

## 1. Federated Learning:

Federated learning (FL), also known as collaborative learning, is a machine learning paradigm where multiple devices, or clients, collaboratively train a shared model without centralizing their data. This approach is particularly advantageous in applications dealing with sensitive data, such as healthcare or finance, as it mitigates privacy and security risks associated with data aggregation.

The FL process typically involves the following steps:

1. A central server distributes an initial model to a subset of clients.
2. Each client trains the model locally on its own data, without sharing the raw data itself.
3. After training, clients send their updated model parameters (e.g., weights and biases) back to the server.
4. The server aggregates these updates from all clients to refine the global model.
5. This iterative process continues for multiple rounds until the desired model performance is achieved.

FL is particularly well-suited for scenarios where data is non-identically distributed (non-IID) across clients, meaning that the data on each client exhibits different statistical properties. This is often the case in real-world applications, where, for example, mobile phone users in different regions may have varying language patterns or app usage. Several types of non-IID data have been identified, including:

- Similar features, different labels
- Similar labels, different features
- Regional and demographically partitioned data
- Client-specific data distributions
- Imbalanced data, a common issue in real-world datasets

Despite its advantages, FL faces several challenges and limitations:

- Data Heterogeneity: Variations in data distributions across clients can negatively impact the performance of the global model.
- Communication Costs: FL necessitates frequent communication between nodes, which can be a bottleneck for devices with limited bandwidth.
- System Heterogeneity: Differences in computational capabilities among client devices can impede the training process.

## 2. FedAvg:

FedAvg is a foundational federated learning algorithm designed to train a shared model by iteratively averaging locally computed model updates from participating clients. The algorithm proceeds as follows:

1. The centralized server initiates a global model with random weights.
2. The server shares this global model with a fraction of selected clients for local model training on their respective datasets.
3. Selected clients update model weights based on their local data, using predefined hyperparameters such as learning rate, batch size, and the number of training epochs.
4. After training, the selected clients send their local model updates back to the server.
5. In each round, the server aggregates the received models, typically through a weighted average (weighted by the number of data points on each client), to update the global model.
6. This process iterates until a predetermined number of communication rounds or a convergence criterion is met.

Advantages of FedAvg:

- Reduced Privacy and Security Risks: By not requiring local raw data to be shared with the centralized server, FL mitigates privacy and security risks associated with data.
- Robustness to Unbalanced and Non-IID Data: Empirical evaluations have demonstrated FedAvg's effectiveness in training models on unbalanced and non-independent data distributions, which are prevalent in real-world scenarios.
- Communication Efficiency: Compared to centralized learning, FL reduces communication between local clients and the server, minimizing communication costs.

Disadvantages of FedAvg:

- Hyperparameter Tuning: Careful tuning of hyperparameters like learning rate, batch size, and the number of training epochs is crucial for optimal performance.
- Model Convergence: Model divergence can occur if hyperparameters are not chosen judiciously. For instance, excessive local training epochs may lead to overfitting.
- System Heterogeneity: Local clients may have varying hardware, network conditions, and data distributions, potentially impacting algorithm performance. For example, clients that do not complete their training process may be excluded, hindering the algorithm's ability to generalize across all data.

Pseudocode:

![](/images/pseudo_code_fedavg.png)

## 3. FedProx:

FedProx is a federated learning optimization algorithm designed to address the challenges of heterogeneity, building upon the earlier FedAvg algorithm to enhance stability and efficiency. Two key ideas underpin the algorithm:

- **Local Proximal Term**: To mitigate divergence and instability arising from variable local updates and statistically heterogeneous data, FedProx incorporates a proximal term into the objective function minimized by each device. The hyperparameter μ controls the proximity of the local model to the global model, akin to a regularization term in other algorithms. This proximal term helps reduce discrepancies, such as biases, between local and global models, thereby improving algorithm stability and convergence.

- **Tolerating Partial Work**: Recognizing the diversity in system capabilities within a federated network, FedProx allows for adaptable workloads tailored to individual client capabilities, unlike FedAvg, which may discard clients that cannot complete all iterations. This flexibility ensures that not all clients need to finish a fixed number of iterations. Depending on their capabilities, some clients may not complete training but can still contribute updates within a communication round. This prevents information loss from these clients, ensuring the model can generalize across all data. This concept is formalized as $\gamma^t_k$-inexactness. This adaptability enables FedProx to accommodate varying levels of local computation across devices and iterations, making it more resilient to system heterogeneity.

Advantages of FedProx:

- **Improved Robustness and Stability**: The proximal term acts as a regularizer, constraining local updates and guiding them towards the global model, leading to improved convergence, particularly in scenarios with significant statistical heterogeneity.

- **Handles Systems Heterogeneity**: FedProx's ability to tolerate partial work makes it well-suited for real-world federated networks where devices have varying computational resources and network connectivity. This allows the model to achieve high performance and generalize to real-world scenarios.

- **Theoretical Convergence Guarantees**: The FedProx paper provides theoretical proof of algorithm convergence under bounded dissimilarity assumptions, which assess the degree of heterogeneity across local clients. Lower dissimilarity metrics indicate higher similarity among clients, ensuring FedProx convergence.

Disadvantages of FedProx:

- **Privacy and Security Risks**: Both algorithms do not inherently guarantee privacy for local samples, as local model weights can still leak information about the training data. Additional measures like differential privacy are needed to ensure data privacy.

- **Tuning Proximal Term**: The hyperparameter $\mu$ requires careful tuning for optimal algorithm performance. A large $\mu$ may slow down convergence by excessively restricting local updates, while a small $\mu$ might not provide sufficient regularization. Although FedProx offers heuristics for setting $\mu$, it still needs to be adjusted for specific problems.

Pseudocode:

![](images/pseudo_code_fedprox.png)

# II. Dataset and Data Partitioning:

## 1. CIFAR-10 Dataset:

The CIFAR-10 dataset is a widely used benchmark dataset for image classification tasks. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. Each image is labeled with one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The CIFAR-10 dataset provides a diverse set of images that allows for the evaluation of various image classification algorithms.

## 2. Data Partitioning Strategy for IID and Non-IID Scenarios:

Initially, the number of clients was fixed at 100, but due to the last client having a significantly higher data density (approximately 10 times), this number was adjusted to 50 for both IID and non-IID datasets.

In the IID scenario, with the defined number of clients, classes, and classes per client, the number of clients selected to receive data for each class is calculated as $M = \frac{\text{clients}}{\text{classes}} \cdot \text{class per client}$. Since each client is intended to have data containing all classes, $M$ should equal the total number of clients. Consequently, the data for each class $i$ is evenly distributed among $\text{clients} - 1$ clients, with the remaining data allocated to the last client. This process is repeated until each client possesses data from all classes.

For the non-IID case, unbalanced data distribution is achieved by calculating $M$ as described above (e.g., for 50 clients, 10 classes, and 2 classes per client, $M = 10$). The number of samples $n_i$ containing class i is then divided among 10 clients. This continues until 10 clients have samples from 2 classes, after which the next 10 clients are considered. The resulting $M$ clients will have 2 consecutive classes, and the subsequent $M'$, $M''$,... groups will be assigned the remaining consecutive class pairs. Notably, to ensure each client has data, the PFLlib repository defines a minimum number of samples as
$\min\left(\frac{\text{batch size}}{1-\text{train ratio}}, \frac{\text{data}}{2 \cdot \text{clients}}\right)$

## 3. Data Partitioning Examples

### 3.1. Example of data partitioning for Non-IID scenario

Run command:

```bash
$cd PFLlib/dataset
$python generate_Cifar10.py noniid - pat 50 Cifar10_niid # The arguments are  the data partitioning strategy, balance or not for partition dataset (- meaning unbalance), number of client and the dataset name
```

Result

```
Number of classes: 10
Client 0	 Size of data: 433	 Labels:  [0 1]
		 Samples of labels:  [(0, 97), (1, 336)]
--------------------------------------------------
Client 1	 Size of data: 609	 Labels:  [0 1]
		 Samples of labels:  [(0, 295), (1, 314)]
--------------------------------------------------
Client 2	 Size of data: 549	 Labels:  [0 1]
		 Samples of labels:  [(0, 132), (1, 417)]
--------------------------------------------------
```

<details>
    <summary>Show more</summary>
    ```
    Client 3	 Size of data: 732	 Labels:  [0 1]
            Samples of labels:  [(0, 204), (1, 528)]
    --------------------------------------------------
    Client 4	 Size of data: 501	 Labels:  [0 1]
            Samples of labels:  [(0, 189), (1, 312)]
    --------------------------------------------------
    Client 5	 Size of data: 1118	 Labels:  [0 1]
            Samples of labels:  [(0, 568), (1, 550)]
    --------------------------------------------------
    Client 6	 Size of data: 908	 Labels:  [0 1]
            Samples of labels:  [(0, 450), (1, 458)]
    --------------------------------------------------
    Client 7	 Size of data: 616	 Labels:  [0 1]
            Samples of labels:  [(0, 341), (1, 275)]
    --------------------------------------------------
    Client 8	 Size of data: 801	 Labels:  [0 1]
            Samples of labels:  [(0, 238), (1, 563)]
    --------------------------------------------------
    Client 9	 Size of data: 5733	 Labels:  [0 1]
            Samples of labels:  [(0, 3486), (1, 2247)]
    --------------------------------------------------
    Client 10	 Size of data: 914	 Labels:  [2 3]
            Samples of labels:  [(2, 538), (3, 376)]
    --------------------------------------------------
    Client 11	 Size of data: 415	 Labels:  [2 3]
            Samples of labels:  [(2, 146), (3, 269)]
    --------------------------------------------------
    Client 12	 Size of data: 525	 Labels:  [2 3]
            Samples of labels:  [(2, 201), (3, 324)]
    --------------------------------------------------
    Client 13	 Size of data: 944	 Labels:  [2 3]
            Samples of labels:  [(2, 453), (3, 491)]
    --------------------------------------------------
    Client 14	 Size of data: 583	 Labels:  [2 3]
            Samples of labels:  [(2, 67), (3, 516)]
    --------------------------------------------------
    Client 15	 Size of data: 510	 Labels:  [2 3]
            Samples of labels:  [(2, 379), (3, 131)]
    --------------------------------------------------
    Client 16	 Size of data: 1041	 Labels:  [2 3]
            Samples of labels:  [(2, 594), (3, 447)]
    --------------------------------------------------
    Client 17	 Size of data: 887	 Labels:  [2 3]
            Samples of labels:  [(2, 373), (3, 514)]
    --------------------------------------------------
    Client 18	 Size of data: 946	 Labels:  [2 3]
            Samples of labels:  [(2, 573), (3, 373)]
    --------------------------------------------------
    Client 19	 Size of data: 5235	 Labels:  [2 3]
            Samples of labels:  [(2, 2676), (3, 2559)]
    --------------------------------------------------
    Client 20	 Size of data: 831	 Labels:  [4 5]
            Samples of labels:  [(4, 575), (5, 256)]
    --------------------------------------------------
    Client 21	 Size of data: 642	 Labels:  [4 5]
            Samples of labels:  [(4, 557), (5, 85)]
    --------------------------------------------------
    Client 22	 Size of data: 530	 Labels:  [4 5]
            Samples of labels:  [(4, 103), (5, 427)]
    --------------------------------------------------
    Client 23	 Size of data: 617	 Labels:  [4 5]
            Samples of labels:  [(4, 86), (5, 531)]
    --------------------------------------------------
    Client 24	 Size of data: 738	 Labels:  [4 5]
            Samples of labels:  [(4, 396), (5, 342)]
    --------------------------------------------------
    Client 25	 Size of data: 439	 Labels:  [4 5]
            Samples of labels:  [(4, 357), (5, 82)]
    --------------------------------------------------
    Client 26	 Size of data: 712	 Labels:  [4 5]
            Samples of labels:  [(4, 526), (5, 186)]
    --------------------------------------------------
    Client 27	 Size of data: 414	 Labels:  [4 5]
            Samples of labels:  [(4, 75), (5, 339)]
    --------------------------------------------------
    Client 28	 Size of data: 565	 Labels:  [4 5]
            Samples of labels:  [(4, 124), (5, 441)]
    --------------------------------------------------
    Client 29	 Size of data: 6512	 Labels:  [4 5]
            Samples of labels:  [(4, 3201), (5, 3311)]
    --------------------------------------------------
    Client 30	 Size of data: 824	 Labels:  [6 7]
            Samples of labels:  [(6, 416), (7, 408)]
    --------------------------------------------------
    Client 31	 Size of data: 465	 Labels:  [6 7]
            Samples of labels:  [(6, 215), (7, 250)]
    --------------------------------------------------
    Client 32	 Size of data: 735	 Labels:  [6 7]
            Samples of labels:  [(6, 373), (7, 362)]
    --------------------------------------------------
    Client 33	 Size of data: 437	 Labels:  [6 7]
            Samples of labels:  [(6, 226), (7, 211)]
    --------------------------------------------------
    Client 34	 Size of data: 729	 Labels:  [6 7]
            Samples of labels:  [(6, 348), (7, 381)]
    --------------------------------------------------
    Client 35	 Size of data: 907	 Labels:  [6 7]
            Samples of labels:  [(6, 478), (7, 429)]
    --------------------------------------------------
    Client 36	 Size of data: 652	 Labels:  [6 7]
            Samples of labels:  [(6, 339), (7, 313)]
    --------------------------------------------------
    Client 37	 Size of data: 668	 Labels:  [6 7]
            Samples of labels:  [(6, 147), (7, 521)]
    --------------------------------------------------
    Client 38	 Size of data: 832	 Labels:  [6 7]
            Samples of labels:  [(6, 303), (7, 529)]
    --------------------------------------------------
    Client 39	 Size of data: 5751	 Labels:  [6 7]
            Samples of labels:  [(6, 3155), (7, 2596)]
    --------------------------------------------------
    Client 40	 Size of data: 1082	 Labels:  [8 9]
            Samples of labels:  [(8, 514), (9, 568)]
    --------------------------------------------------
    Client 41	 Size of data: 844	 Labels:  [8 9]
            Samples of labels:  [(8, 574), (9, 270)]
    --------------------------------------------------
    Client 42	 Size of data: 365	 Labels:  [8 9]
            Samples of labels:  [(8, 209), (9, 156)]
    --------------------------------------------------
    Client 43	 Size of data: 652	 Labels:  [8 9]
            Samples of labels:  [(8, 323), (9, 329)]
    --------------------------------------------------
    Client 44	 Size of data: 207	 Labels:  [8 9]
            Samples of labels:  [(8, 137), (9, 70)]
    --------------------------------------------------
    Client 45	 Size of data: 474	 Labels:  [8 9]
            Samples of labels:  [(8, 135), (9, 339)]
    --------------------------------------------------
    Client 46	 Size of data: 604	 Labels:  [8 9]
            Samples of labels:  [(8, 392), (9, 212)]
    --------------------------------------------------
    Client 47	 Size of data: 639	 Labels:  [8 9]
            Samples of labels:  [(8, 103), (9, 536)]
    --------------------------------------------------
    Client 48	 Size of data: 1068	 Labels:  [8 9]
            Samples of labels:  [(8, 592), (9, 476)]
    --------------------------------------------------
    Client 49	 Size of data: 6065	 Labels:  [8 9]
            Samples of labels:  [(8, 3021), (9, 3044)]
    --------------------------------------------------
    Total number of samples: 60000
    The number of train samples: [324, 456, 411, 549, 375, 838, 681, 462, 600, 4299, 685, 311, 393, 708, 437, 382, 780, 665, 709, 3926, 623, 481, 397, 462, 553, 329, 534, 310, 423, 4884, 618, 348, 551, 327, 546, 680, 489, 501, 624, 4313, 811, 633, 273, 489, 155, 355, 453, 479, 801, 4548]
    The number of test samples: [109, 153, 138, 183, 126, 280, 227, 154, 201, 1434, 229, 104, 132, 236, 146, 128, 261, 222, 237, 1309, 208, 161, 133, 155, 185, 110, 178, 104, 142, 1628, 206, 117, 184, 110, 183, 227, 163, 167, 208, 1438, 271, 211, 92, 163, 52, 119, 151, 160, 267, 1517]
    ```
</details>

### 3.2. Example of data partitioning for IID scenario

Run command:

```bash
$cd PFLlib/dataset
$python generate_Cifar10.py iid balance - 50 Cifar10_iid # The arguments are  the data partitioning strategy, balance or not for partition dataset, partition when option is unbalance, number of client and the dataset name
```

Result:

```
Number of classes: 10
Client 0	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
		 Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
--------------------------------------------------
Client 1	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
		 Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
--------------------------------------------------
Client 2	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
		 Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
--------------------------------------------------
```

<details>
    <summary>Show more</summary>

    Client 3	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 4	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 5	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 6	 Size of data: 12**Theoretical Convergence Guarantees**00	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 7	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 8	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 9	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 10	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 11	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    -----------------------------**Theoretical Convergence Guarantees**---------------------
    Client 12	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 13	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 14	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 15	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 16	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 17	 Size of data: 1**Theoretical Convergence Guarantees**----------------------
    Client 18	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 19	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 20	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 21	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 22	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    ----------------------------**Theoretical Convergence Guarantees**----------------------
    Client 23	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 24	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 25	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 26	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 27	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 28	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), **Theoretical Convergence Guarantees**(3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 29	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 30	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 31	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 32	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 33	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  **Theoretical Convergence Guarantees**[(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 34	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, **Theoretical Convergence Guarantees**120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 35	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 36	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 37	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 38	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 39	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 40	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 41	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 42	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 43	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 44	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 45	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 46	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 47	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 48	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Client 49	 Size of data: 1200	 Labels:  [0 1 2 3 4 5 6 7 8 9]
            Samples of labels:  [(0, 120), (1, 120), (2, 120), (3, 120), (4, 120), (5, 120), (6, 120), (7, 120), (8, 120), (9, 120)]
    --------------------------------------------------
    Total number of samples: 60000
    The number of train samples: [900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900]
    The number of test samples: [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]

</details>

# III. Model Architecture

Utilizing the CIFAR-10 dataset, I employed a fundamental CNN architecture as suggested in the FedAvg paper. This architecture comprises two convolutional layers with 5x5 filters, containing 32 and 64 channels respectively, each followed by a 2x2 max pooling layer. Subsequently, a fully connected layer with 512 units and ReLU activation is implemented, culminating in a softmax layer with 10 classes (totaling 1,663,370 parameters). Although the FedAvg paper proposed transformation techniques such as cropping to 24x24, random horizontal flipping, and adjusting contrast, brightness, and whitening, these did not yield the desired performance improvement compared to simple pixel normalization as suggested in the PFLlib repository. Consequently, the transformations from the library were retained.

# IV. Federated Learning Setup

In this experimental study, the PFLlib repository was utilized for evaluation and comparison, chosen partly due to recommendations and its comprehensive handling of functionalities required for experiments based on the FedAvg and FedProx papers. To facilitate the experimental process, certain modifications were made to the original PFLlib repository, streamlining the training and evaluation procedures. While certain hyper-parameters were adopted from these papers and kept constant, others were slightly adjusted to meet specific requirements:

- Number of clients (K): 50 clients were selected as this represents a substantial number suitable for the CIFAR-10 dataset, with less sample distribution variance compared to 100 clients where the last client has 10 times fewer local samples than the preceding clients.
- Communication rounds (T): 200 rounds of communication between the server and clients were deemed reasonable for multiple evaluations while saving computational time.
- Batch size (B): A mini-batch size of 20 was arbitrarily chosen.
- Learning rate: A fixed learning rate of 0.05 was chosen based on literature review and to save computational resources compared to experimenting with learning rate decay.
- Learning rate decay: A gamma value of 0.99 was used for learning rate decay after each round, compared against a fixed learning rate.
- Max local epochs (E): 5 local epochs were found to yield better results compared to 1 or 10 as suggested in the papers.
- Join ratio (C): A drop rate of 0.3 was selected for evaluating system heterogeneity, ensuring sufficient active clients (e.g., C=0.3, K=50 with 90% drop-rate results in 13 active clients) to update the global model even with a 90% straggler drop rate.
- Algorithm: FedAvg and FedProx aggregation methods were employed to compare their performance. For FedProx, the hyperparameter mu was varied within [0.001, 0.01, 0.1, 1] to assess if increasing mu slows down global model convergence.
- Data partitioning: Data was partitioned into IID and non-IID scenarios.
- Client Drop Rate: Drop rates of [0.0, 0.5, 0.9] were applied to evaluate model performance under client dropouts and the partial work tolerance of FedProx as discussed in the paper.

# V. Results: Present the obtained test accuracy for FedAvg and FedProx under both data distribution scenarios, preferably with visualizations (e.g., plots showing accuracy over communication rounds).

<table>
<caption><b>Table 1: Comparison of FedAvg and FedProx Algorithms Performance on IID and Non-IID Datasets with Varying Drop Rates and Proximal Term (μ) with Fraction of clients (C) is 30%</b></caption> 
<thead>
<tr>
<th>Algorithm</th>
<th>Dataset Type</th>
<th>Drop Rate (%)</th>
<th>μ</th>
<th>Best Accuracy (%)</th>
<th>Average Time Cost per Round (s)</th>
<th>Average Time Cost</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6">FedAvg</td>
<td rowspan="3">IID</td>
<td>0</td>
<td>-</td>
<td>64.44</td>
<td>12.03s</td>
<td>40.34 min</td>
</tr>
<tr>
<td>50</td>
<td></td>
<td>64.01</td>
<td>12.25s</td>
<td>41.09 min</td>
</tr>
<tr>
<td>90</td>
<td></td>
<td>63.42</td>
<td>24.86s</td>
<td>1.39 hours</td>
</tr>
<tr>
<td rowspan="3">Non-IID</td>
<td>0</td>
<td></td>
<td>61.92</td>
<td>11.96s</td>
<td>40.08 min</td>
</tr>
<tr>
<td>50</td>
<td></td>
<td>61.57</td>
<td>14.29s</td>
<td>47.9 min</td>
</tr>
<tr>
<td>90</td>
<td></td>
<td>38.56</td>
<td>13.07s</td>
<td>43.93 min</td>
</tr>
<tr>
<td rowspan="12">FedProx</td>
<td rowspan="6">IID</td>
<td>0</td>
<td>0.001</td>
<td>64.32</td>
<td>14.88s</td>
<td>49.92 min</td>
</tr>
<tr>
<td>50</td>
<td></td>
<td>63.79</td>
<td>14.74s</td>
<td>49.41 min</td>
</tr>
<tr>
<td>90</td>
<td></td>
<td>62.3</td>
<td>15.31s</td>
<td>51.33 min</td>
</tr>
<tr>
<td>0</td>
<td>0</td>
<td>64.03</td>
<td>15.17s</td>
<td>50.89 min</td>
</tr>
<tr>
<td>50</td>
<td></td>
<td>63.73</td>
<td>15.45s</td>
<td>51.8 min</td>
</tr>
<tr>
<td>90</td>
<td></td>
<td>62.71</td>
<td>15.49s</td>
<td>51.95 min</td>
</tr>
<tr>
<td rowspan="6">Non-IID</td>
<td>0</td>
<td>0.001</td>
<td>63.35</td>
<td>15.91s</td>
<td>53.29 min</td>
</tr>
<tr>
<td>50</td>
<td></td>
<td>60.71</td>
<td>15.15s</td>
<td>50.75 min</td>
</tr>
<tr>
<td>90</td>
<td></td>
<td>40.36</td>
<td>14.9s</td>
<td>49.93 min</td>
</tr>
<tr>
<td>0</td>
<td>0</td>
<td>61.62</td>
<td>15.59s</td>
<td>52.3 min</td>
</tr>
<tr>
<td>50</td>
<td></td>
<td>59.11</td>
<td>16.03s</td>
<td>53.69 min</td>
</tr>
<tr>
<td>90</td>
<td></td>
<td>32.85</td>
<td>14.33s</td>
<td>48.24 min</td>
</tr>
</tbody>
</table>

<figure>
    <img src="images/hinh1_full.png" alt="Comparison of FedAvg and FedProx performance under different straggler conditions and dataset distributions" style="width: 1000px; height: auto;"/>
    <figcaption>Figure 1: Testing accuracy of FedAvg and FedProx (with and without the proximal term) on I.I.D. and non-I.I.D. datasets under various percentages of stragglers (0%, 50%, and 90%). The x-axis represents the number of communication rounds, and the y-axis represents testing accuracy.</figcaption>
</figure>

<figure>
    <img src="images/hinh2_full.png" alt="Comparison of Training Loss for FedAvg and FedProx Algorithms on IID and Non-IID Datasets under Varying Straggler Percentages" style="width: 1000px; height: auto;" /> 
    <figcaption>Figure 2: Impact of stragglers on training loss for FedAvg and FedProx across I.I.D. and non-I.I.D. datasets.</figcaption>
</figure>

<figure>
    <img src="images/hinh3_full.png" alt="Training loss of FedProx with non-IID data for various mu candidates" style="width: 2000px; height: auto;">
    <figcaption>Figure 3: Training loss and Test Accuracy of FedProx with non-IID data for various μ (proximal term) candidates.</figcaption>
</figure>


# VI. Analysis and Discussion: 

## 1. System Heterogeneity and Data Distribution Impact on Federated Learning Performance

In this experiment, we investigated the impact of varying levels of system heterogeneity on federated learning performance by adjusting the proportion of dropped devices (stragglers) to 0%, 50%, and 90%. As highlighted in the FedProx paper, the FedAvg algorithm discards stragglers upon reaching the global clock cycle, potentially discarding valuable information from incomplete works on devices. In contrast, FedProx incorporates partial updates from these straggler devices, enhancing the global model update.

Our analysis of the IID dataset with no stragglers revealed that all three algorithms (FedAvg, FedProx ($\mu = 0$), and FedProx ($\mu > 0$)) rapidly achieve high testing accuracy, with FedProx ($\mu > 0$) exhibiting slightly better stability. For the non-IID dataset with no stragglers (0% drop-rate), FedAvg and FedProx ($\mu = 0$) demonstrate significant fluctuations in accuracy, indicating instability, while FedProx ($\mu > 0$) illustrated improved stability and higher bit accuracy.

As the percentage of stragglers increased (50% and 90% of drop rate), the performance of FedAvg and FedProx ($\mu = 0$) degenerated, particularly in the non-IID scenario. With 50% stragglers, both algorithms exhibited slower convergence and increased instability compared to the 0% straggler case, while FedProx ($\mu > 0$) maintained superior performance in terms of stability, convergence speed, and accuracy. This trend persisted with 90% stragglers, where FedAvg and FedProx ($\mu = 0$) struggled significantly with the non-IID dataset, while FedProx ($\mu > 0$) showed greater robustness, albeit with lower overall accuracy. Notably, FedProx ($\mu > 0$) consistently outperformed FedAvg by approximately 2% on average in highly heterogeneous environments (90% stragglers).

The training loss analysis further supports these findings. In the IID dataset with 0% stragglers, all algorithms demonstrated rapid convergence and low training loss. However, for the non-IID dataset illustrated in plots, FedAvg and FedProx (μ = 0) exhibited higher fluctuations, indicating instability, while FedProx ($\mu > 0$) maintained relatively stable and lower training loss. With increasing straggler percentages, FedAvg and FedProx ($\mu = 0$) showed slower training loss reduction and increased fluctuations, especially in the non-IID scenario. Although FedProx ($\mu > 0$) experienced similar trends, it consistently achieved lower and kept less fluctuation training loss compared to the other algorithms, highlighting its robustness in handling high straggler percentages and non-IID data.

In conclusion, this comprehensive analysis demonstrates that while FedAvg performs adequately with IID data and no stragglers, it struggles with non-IID data and high straggler percentages in showcases. FedProx ($\mu = 0$) mirrors this pattern, suggesting that the proximal term is generally crucial for mitigating the negative effects of system heterogeneity. Conversely, FedProx ($\mu > 0$) is consistently better than the other algorithms, particularly in challenging scenarios with non-IID data and high straggler percentages, underscoring the proximal term's effectiveness in enhancing stability and convergence.


## 2. Tuning the Proximal Term (μ) in FedProx

With the penalty constant μ fixed at 0.001, the question arises whether this value is suitable for achieving convergence in a system heterogeneous setting with non-IID Cifar-10 dataset. While an excessively large $\mu$ can slow down convergence and restrict weight updates to near their initial values, a small \$mu$ may render the proximal term ineffective. In this experiment, we propose to determine the optimal μ from the candidate set $\{0.001, 0.01, 0.1, 1\}$. As shown in Figure 3 and discussed in the FedProx paper, larger μ values lead to slower convergence, albeit with improved stability as observed in the training plots. Conversely, for $0 < \mu \leq 0.01$, despite fluctuations and the need for more communication rounds to converge, the results suggest that selecting μ within this range yields promising outcomes for this specific problem.


# VII. Limitations
A limitation of this experimental study is the absence of a dissimilarity metric to quantify the differences between local devices. Such a metric would provide a more intuitive understanding of why FedProx with μ > 0 struggles to converge in the non-IID data setting as the level of system heterogeneity increases.


# VIII. Instructions to run the code

This experimental study is based on the open-source code from the PFLlib repository, a Python library for distributed federated learning. To reproduce the results, please follow the steps outlined in the PFLlib repository documentation as below.    

## 1. Install the required dependencies

Install [CUDA](https://developer.nvidia.com/cuda-downloads).

Install [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda.

Create a new conda environment using the provided environment file:

```bash
$cd PFLlib
$conda env create -f env_cuda_latest.yaml # You may need to downgrade the torch using pip to match the CUDA version
```

## 2. How to start simulating

- Download [this project](https://github.com/kisejin/FL_fundamental) to an appropriate location using [git](https://git-scm.com/).

  ```bash
  $git clone https://github.com/kisejin/FL_fundamental.git
  ```

- Build evaluation scenarios (see [Data Partitioning Examples](#3-data-partitioning-examples)).

- Activate the conda environment:

  ```bash
  $conda activate pfllib
  ```

- Run evaluation example for FedAvg:

  ```bash
  $source run_fedavg.sh Cifar10_niid 0.3 0.9 # The arguments are ordered as dataset_type, fraction client ratio, and drop rate
  ```

- Run evaluation example for FedProx:

  ```bash
  $source run_fedprox.sh Cifar10_niid 0.3 0.9 0.001 # The arguments are ordered as dataset_type, fraction client ratio, drop rate, mu
  ```

- The arguments I used in the example are as follows:

  - dataset_type: `Cifar10_iid` (IID), `Cifar10_niid` (non-IID)
  - fraction client ratio: `0.3` (30%)
  - drop rate: [`0.0`, `0.5`, `0.9`] active clients
  - mu: fixed `0.001` for FedProx

- As long hyper-parameters are set, so I compress it into the `run_fedavg.sh` and `run_fedprox.sh` files. You can modify these files to change the hyper-parameters. Detail training configurations can be found in the `Run_FL.ipynb` file.

- Show the results by plotting the figures in the `plot_results.py` file.
   ```bash
   $python plot_results.py --type_plot plot_drop_rate --mode test_acc --output_name test_acc_drop_rate
   ``` 
   The detail the configuration arguments can be found in the `plot_results.py` file.
# Citation

```
@inproceedings{mcmahan2017communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  booktitle={Artificial intelligence and statistics},
  pages={1273--1282},
  year={2017},
  organization={PMLR}
}

@article{li2020federated,
  title={Federated optimization in heterogeneous networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  journal={Proceedings of Machine learning and systems},
  volume={2},
  pages={429--450},
  year={2020}
}
```
