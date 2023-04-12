from dataset import Lorenz63, SantaFe
import numpy as np
import matplotlib.pyplot as plt


class NGRCSolver(object):
    def __init__(self, dataset, settings):
        self.dataset = dataset
        self.settings = settings
        self.task = dataset.name
        self.data = dataset.data
        self.symbol = None
        self.normalize_data = dict({"normed": settings["normed"]})
        self.reconstruct_data()
        self.results = {
            "training_data": np.NAN,
            "training_ground_truth": np.NAN,
            "inferring_data": np.NAN,
            "inferring_ground_truth": np.NAN,
            "nmse": np.NAN
        }

    def reconstruct_data(self):
        self.data = self.data.reshape(1, -1, order='F')

    def get_feature_vector(self, matrix):
        length = matrix.shape[0]
        d = self.dataset.dimension
        k = self.settings["time_lag"]

        linear = d * k
        nonlinear = round(linear * (linear + 1) / 2)
        total = 1 + linear + nonlinear

        feature_vector = np.ones([total, length])
        for i in range(k):
            feature_vector[1 + i * d: 1 + i * d + d, :] = matrix[:, d * k - i * d - d:d * k - i * d].T

        cnt = 0
        for i in range(linear):
            for j in range(i, linear):
                feature_vector[linear + 1 + cnt] = feature_vector[1 + i] * feature_vector[1 + j]
                cnt += 1

        return feature_vector

    def train(self, training_data, training_ground_truth):
        yd = training_ground_truth.T
        fv = self.get_feature_vector(training_data)
        ridge_param = self.settings["ridge_param"]
        solver = self.settings["wout_solver"]
        wout = None
        if solver == "old1":
            wout = yd @ fv.T @ np.linalg.inv(fv @ fv.T + ridge_param * np.identity(fv.shape[0]))
        elif solver == "old2":
            wout = yd @ fv.T @ np.linalg.pinv(fv @ fv.T + ridge_param * np.identity(fv.shape[0]))
        elif solver == "new1":
            s1 = np.diag(1 / np.max(fv, axis=1))
            sr = s1 @ fv
            wout = yd @ sr.T @ np.linalg.inv(sr @ sr.T) @ s1
        elif solver == "new2":
            s1 = np.diag(1 / np.max(fv, axis=1))
            sr = s1 @ fv
            wout = yd @ sr.T @ np.linalg.inv(sr @ sr.T) @ s1
        return wout

    @staticmethod
    def get_nmse(target, predict):
        return np.sum((target - predict) ** 2) / np.size(predict) / np.var(target)

    def infer(self, wout, inferring_data):
        fv = self.get_feature_vector(inferring_data)
        return (wout @ fv).T

    def matrix_transfer(self, matrix):
        dimension = self.dataset.dimension
        time_lag = self.settings["time_lag"]
        length = int(matrix.size/dimension-time_lag+1)
        new_matrix = np.zeros([length, dimension * time_lag])
        for line in range(length):
            new_matrix[line, :] = matrix[line * dimension: line * dimension + dimension * time_lag]
        return new_matrix

    def matrix_transfer_ground_truth(self, matrix):
        dimension = self.dataset.dimension
        length = int(matrix.size/dimension)
        new_matrix = np.zeros([length, dimension])
        for line in range(length):
            new_matrix[line, :] = matrix[line * dimension: line * dimension + dimension]
        return new_matrix

    def normalize(self):
        max_value = self.normalize_data["max_value"] = self.data.max()
        min_value = self.normalize_data["min_value"] = self.data.min()
        self.data = (self.data - min_value) / (max_value - min_value) * 0.5 + 0.5
        if self.dataset.name == "Equalisation":
            self.symbol = (self.dataset.symbol - min_value) / (max_value - min_value) * 0.5 + 0.5

    def inverse_normalize(self, data):
        max_value = self.normalize_data["max_value"]
        min_value = self.normalize_data["min_value"]
        return (data - 0.5) * 2 * (max_value - min_value) + min_value

    @staticmethod
    def judge_symbol(data):
        data[data < -2] = -3
        data[(data < 0) & (data > -2)] = -1
        data[(data < 2) & (data > 0)] = 1
        data[data > 2] = 3

    def run(self):
        if self.normalize_data["normed"]:
            self.normalize()

        start = self.settings["start"]
        length = self.settings["length"]
        ratio = self.settings["ratio"]
        d = self.dataset.dimension
        k1 = self.settings["time_lag"] * self.dataset.dimension
        k2 = d
        train_start = start * d
        train_end = (start + round(length * ratio) + self.settings["time_lag"] - 1) * d
        infer_start = (start + round(length * ratio)) * d
        infer_end = (start + length + self.settings["time_lag"] - 1) * d
        train_gt_start = train_start + self.settings["time_lag"] * d
        train_gt_end = train_end + d
        infer_gt_start = infer_start + self.settings["time_lag"] * d
        infer_gt_end = infer_end + d
        # print(
        #     "training data start: {}\n".format(train_start) +
        #     "training data end: {}\n".format(train_end) +
        #     "training ground truth start: {}\n".format(train_gt_start) +
        #     "training ground truth end: {}\n".format(train_gt_end) +
        #     "inferring data start: {}\n".format(infer_start) +
        #     "inferring data end: {}\n".format(infer_end) +
        #     "inferring ground truth start: {}\n".format(infer_gt_start) +
        #     "inferring ground truth end: {}\n".format(infer_gt_end)
        # )

        if self.dataset.name == "SantaFe":
            training_data = self.matrix_transfer(self.data[0, train_start: train_end])
            inferring_data = self.matrix_transfer(self.data[0, infer_start: infer_end])
            training_ground_truth = self.matrix_transfer_ground_truth(
                self.data[0, train_gt_start: train_gt_end])
            inferring_ground_truth = self.matrix_transfer_ground_truth(
                self.data[0, infer_gt_start: infer_gt_end])

            wout = self.train(training_data, training_ground_truth)
            predict = self.infer(wout, inferring_data)
            if self.normalize_data["normed"]:
                predict = self.inverse_normalize(predict)
                inferring_ground_truth = self.inverse_normalize(inferring_ground_truth)
                nmse = self.get_nmse(inferring_ground_truth, predict)
            else:
                nmse = self.get_nmse(inferring_ground_truth, predict)
            self.results["predict"] = predict
            self.results["training_data"] = training_data
            self.results["training_ground_truth"] = training_ground_truth
            self.results["inferring_data"] = inferring_data
            self.results["inferring_ground_truth"] = inferring_ground_truth
            self.results["nmse"] = nmse
            print(self.results["nmse"])
            self.visualize()

        elif self.dataset.name == "Equalisation":
            self.symbol = self.dataset.symbol
            train_start = start * d
            train_end = (start + round(length * ratio) + self.settings["time_lag"] - 1) * d
            infer_start = (start + round(length * ratio)) * d
            infer_end = (start + length + self.settings["time_lag"] - 1) * d
            train_gt_start = train_start + self.settings["time_lag"] * d - d
            train_gt_end = train_end
            infer_gt_start = infer_start + self.settings["time_lag"] * d - d
            infer_gt_end = infer_end
            training_data = self.matrix_transfer(self.data[0, train_start: train_end])
            inferring_data = self.matrix_transfer(self.data[0, infer_start: infer_end])
            training_ground_truth = self.matrix_transfer_ground_truth(
                self.symbol[0, train_gt_start: train_gt_end])
            inferring_ground_truth = self.matrix_transfer_ground_truth(
                self.symbol[0, infer_gt_start: infer_gt_end])
            training_origin = self.matrix_transfer_ground_truth(
                self.data[0, train_gt_start: train_gt_end])

            wout = self.train(training_data, training_ground_truth)
            predict = self.infer(wout, inferring_data)
            if self.normalize_data["normed"]:
                predict = self.inverse_normalize(predict)
                training_origin = self.inverse_normalize(training_data)
                self.judge_symbol(predict)
                self.judge_symbol(training_origin)
                inferring_ground_truth = self.inverse_normalize(inferring_ground_truth)
                nmse = self.get_nmse(inferring_ground_truth, predict)
            else:
                nmse = self.get_nmse(inferring_ground_truth, predict)
                self.judge_symbol(predict)
            self.results["predict"] = predict
            self.results["training_data"] = training_data
            self.results["training_ground_truth"] = training_ground_truth
            self.results["inferring_data"] = inferring_data
            self.results["inferring_ground_truth"] = inferring_ground_truth
            self.results["nmse"] = nmse
            print(self.results["nmse"])

            temp = self.results["training_ground_truth"] - training_origin
            temp[temp < 0.1] = 0
            print("(train) The number of symbol error: {}".format(np.count_nonzero(temp)))

            temp = self.results["inferring_ground_truth"] - self.results["predict"]
            temp[temp < 0.1] = 0
            print("(infer) The number of symbol error: {}".format(np.count_nonzero(temp)))
            self.visualize()

        elif self.dataset.name == "Lorenz63":
            training_data = self.matrix_transfer(self.data[0, train_start: train_end])
            inferring_data = self.matrix_transfer(self.data[0, infer_start: infer_end])
            training_ground_truth = self.matrix_transfer_ground_truth(
                self.data[0, train_gt_start: train_gt_end])
            inferring_ground_truth = self.matrix_transfer_ground_truth(
                self.data[0, infer_gt_start: infer_gt_end])
            # print(training_data.shape)
            # print(inferring_data.shape)
            # print(training_ground_truth.shape)
            # print(inferring_ground_truth.shape)
            print(training_data.shape)
            print(training_ground_truth.shape)
            print(inferring_data.shape)
            print(inferring_ground_truth.shape)
            wout = self.train(training_data, training_ground_truth)
            predict = self.infer(wout, inferring_data)
            print(predict.shape)
            if self.normalize_data["normed"]:
                predict = self.inverse_normalize(predict)
                inferring_ground_truth = self.inverse_normalize(inferring_ground_truth)
                nmse = self.get_nmse(inferring_ground_truth, predict)
            else:
                nmse = self.get_nmse(inferring_ground_truth, predict)
            self.results["predict"] = predict
            self.results["training_data"] = training_data
            self.results["training_ground_truth"] = training_ground_truth
            self.results["inferring_data"] = inferring_data
            self.results["inferring_ground_truth"] = inferring_ground_truth
            self.results["nmse"] = nmse
            print(self.results["nmse"])
            self.visualize()

    def visualize(self):
        if self.dataset.name == "SantaFe":
            target = self.results["inferring_ground_truth"]
            predict = self.results["predict"]
            fig, ax = plt.subplots()
            x = np.arange(target.shape[0])
            ax.plot(x, target, label="ground truth")
            ax.plot(x, predict, label="predict")
            ax.set_xlim(0, 200)
            ax.legend()
            plt.show()
        elif self.dataset.name == "Equalisation":
            target = self.results["inferring_ground_truth"]
            predict = self.results["predict"]
            fig, ax = plt.subplots()
            x = np.arange(target.shape[0])
            ax.step(x, target, label="ground truth")
            ax.step(x, predict, label="predict")
            ax.set_xlim(0, 50)
            ax.legend()
            plt.show()
        elif self.dataset.name == "Lorenz63":
            training_data = self.results["training_data"].T
            target = self.results["inferring_ground_truth"].T
            predict = self.results["predict"].T
            t_linewidth = 1.1
            a_linewidth = 0.3
            plt.rcParams.update({'font.size': 12})

            fig1 = plt.figure()
            fig1.set_figheight(8)
            fig1.set_figwidth(12)

            xlabel = [10, 15, 20, 25, 30]
            h = 120
            w = 100

            axs1 = plt.subplot2grid(shape=(h, w), loc=(0, 9), colspan=22, rowspan=38)
            axs2 = plt.subplot2grid(shape=(h, w), loc=(52, 0), colspan=42, rowspan=20)
            axs3 = plt.subplot2grid(shape=(h, w), loc=(75, 0), colspan=42, rowspan=20)
            axs4 = plt.subplot2grid(shape=(h, w), loc=(98, 0), colspan=42, rowspan=20)
            axs5 = plt.subplot2grid(shape=(h, w), loc=(0, 61), colspan=22, rowspan=38)
            axs6 = plt.subplot2grid(shape=(h, w), loc=(52, 50), colspan=42, rowspan=20)
            axs7 = plt.subplot2grid(shape=(h, w), loc=(75, 50), colspan=42, rowspan=20)
            axs8 = plt.subplot2grid(shape=(h, w), loc=(98, 50), colspan=42, rowspan=20)

            axs1.plot(training_data[0, :], training_data[2, :],
                      linewidth=a_linewidth)
            axs1.set_xlabel('x')
            axs1.set_ylabel('z')
            axs1.set_title('ground truth')
            # axs1.axes.set_xbound(-21, 21)
            # axs1.axes.set_ybound(2, 48)

            axs2.set_title('training phase')
            axs2.plot(target[0, :],
                      linewidth=t_linewidth)
            axs2.set_ylabel('x')
            axs2.axes.xaxis.set_ticklabels([])
            # axs2.axes.set_ybound(-21, 21)
            # axs2.axes.set_xbound(-.15, 10.15)

            axs3.plot(target[1, :],
                      linewidth=t_linewidth)
            axs3.set_ylabel('y')
            axs3.axes.xaxis.set_ticklabels([])
            # axs3.axes.set_xbound(-.15, 10.15)

            axs4.plot(target[2, :],
                      linewidth=t_linewidth)
            axs4.set_ylabel('y')
            axs4.set_xlabel('time')
            # axs4.axes.set_xbound(-.15, 10.15)

            axs5.plot(predict[0, :], predict[2, :],
                      linewidth=a_linewidth)
            axs5.set_xlabel('x')
            axs5.set_ylabel('z')
            axs5.set_title('NGRC prediction')
            # axs5.axes.set_xbound(-21, 21)
            # axs5.axes.set_ybound(2, 48)

            axs6.set_title('training phase')
            axs6.plot(predict[0, :],
                      linewidth=t_linewidth)
            axs6.set_ylabel('x')
            axs6.axes.xaxis.set_ticklabels([])
            # axs6.axes.set_ybound(-21, 21)
            # axs6.axes.set_xbound(-.15, 10.15)

            axs7.plot(predict[1, :],
                      linewidth=t_linewidth)
            axs7.set_ylabel('y')
            axs7.axes.xaxis.set_ticklabels([])
            # axs7.axes.set_xbound(-.15, 10.15)

            axs8.plot(predict[2, :],
                      linewidth=t_linewidth)
            axs8.set_ylabel('y')
            axs8.set_xlabel('time')
            # axs8.axes.set_xbound(-.15, 10.15)

            plt.show()

    # def visualize1(self, target, predict):
    #     t_linewidth = 1.1
    #     a_linewidth = 0.3
    #     plt.rcParams.update({'font.size': 12})
    #
    #     fig1 = plt.figure()
    #     fig1.set_figheight(8)
    #     fig1.set_figwidth(12)
    #
    #     xlabel = [10, 15, 20, 25, 30]
    #     h = 120
    #     w = 100
    #
    #     axs1 = plt.subplot2grid(shape=(h, w), loc=(0, 9), colspan=22, rowspan=38)
    #     axs2 = plt.subplot2grid(shape=(h, w), loc=(52, 0), colspan=42, rowspan=20)
    #     axs3 = plt.subplot2grid(shape=(h, w), loc=(75, 0), colspan=42, rowspan=20)
    #     axs4 = plt.subplot2grid(shape=(h, w), loc=(98, 0), colspan=42, rowspan=20)
    #     axs5 = plt.subplot2grid(shape=(h, w), loc=(0, 61), colspan=22, rowspan=38)
    #     axs6 = plt.subplot2grid(shape=(h, w), loc=(52, 50), colspan=42, rowspan=20)
    #     axs7 = plt.subplot2grid(shape=(h, w), loc=(75, 50), colspan=42, rowspan=20)
    #     axs8 = plt.subplot2grid(shape=(h, w), loc=(98, 50), colspan=42, rowspan=20)
    #
    #     axs1.plot(self.data[0, warmup_pts:total_pts], self.data[2, warmup_pts:total_pts],
    #               linewidth=a_linewidth)
    #     axs1.set_xlabel('x')
    #     axs1.set_ylabel('z')
    #     axs1.set_title('ground truth')
    #     axs1.axes.set_xbound(-21, 21)
    #     axs1.axes.set_ybound(2, 48)
    #
    #     axs2.set_title('training phase')
    #     axs2.plot(t_eval[warmup_pts:warmup_pts + train_pts]-warmup, self.data[0, warmup_pts:warmup_pts + train_pts],
    #               linewidth=t_linewidth)
    #     axs2.set_ylabel('x')
    #     axs2.axes.xaxis.set_ticklabels([])
    #     axs2.axes.set_ybound(-21, 21)
    #     axs2.axes.set_xbound(-.15, 10.15)
    #
    #     axs3.plot(t_eval[warmup_pts:warmup_pts + train_pts]-warmup, self.data[1, warmup_pts:warmup_pts + train_pts],
    #               linewidth=t_linewidth)
    #     axs3.set_ylabel('y')
    #     axs3.axes.xaxis.set_ticklabels([])
    #     axs3.axes.set_xbound(-.15, 10.15)
    #
    #     axs4.plot(t_eval[warmup_pts:warmup_pts + train_pts]-warmup, self.data[2, warmup_pts:warmup_pts + train_pts],
    #               linewidth=t_linewidth)
    #     axs4.set_ylabel('y')
    #     axs4.set_xlabel('time')
    #     axs4.axes.set_xbound(-.15, 10.15)
    #
    #     axs5.plot(self.data[0, warmup_pts:total_pts], self.data[2, warmup_pts:total_pts],
    #               linewidth=a_linewidth)
    #     axs5.set_xlabel('x')
    #     axs5.set_ylabel('z')
    #     axs5.set_title('NGRC prediction')
    #     axs5.axes.set_xbound(-21, 21)
    #     axs5.axes.set_ybound(2, 48)
    #
    #     axs6.set_title('training phase')
    #     axs6.plot(t_eval[warmup_pts:warmup_pts + train_pts]-warmup, self.data[0, warmup_pts:warmup_pts + train_pts],
    #               linewidth=t_linewidth)
    #     axs6.set_ylabel('x')
    #     axs6.axes.xaxis.set_ticklabels([])
    #     axs6.axes.set_ybound(-21, 21)
    #     axs6.axes.set_xbound(-.15, 10.15)
    #
    #     axs7.plot(t_eval[warmup_pts:warmup_pts + train_pts]-warmup, self.data[1, warmup_pts:warmup_pts + train_pts],
    #               linewidth=t_linewidth)
    #     axs7.set_ylabel('y')
    #     axs7.axes.xaxis.set_ticklabels([])
    #     axs7.axes.set_xbound(-.15, 10.15)
    #
    #     axs8.plot(t_eval[warmup_pts:warmup_pts + train_pts]-warmup, self.data[2, warmup_pts:warmup_pts + train_pts],
    #               linewidth=t_linewidth)
    #     axs8.set_ylabel('y')
    #     axs8.set_xlabel('time')
    #     axs8.axes.set_xbound(-.15, 10.15)
    #
    #     plt.show()

    def print_information(self):
        print(
            "#######################################################\n" +
            "Task: {}\n".format(self.dataset.name) +
            "DataSet Dimension: {}\n".format(self.dataset.dimension) +
            "DataSet Total Size: {}\n".format(self.dataset.length) +
            "#######################################################\n" +
            "Data Information: \n" +
            "From {}".format(self.ngrc_settings["start"]) +
            " to {}, ".format(self.ngrc_settings["start"] + self.ngrc_settings["length"]) +
            "the first {:.2f}% are used for training ".format(self.ngrc_settings["ratio"] * 100) +
            "and the remaining {:.2f}% are used for testing.\n".format((1 - self.ngrc_settings["ratio"]) * 100) +
            "Is Normalized: {}\n".format("Yes" if self.ngrc_settings["normed"] else "No") +
            "Time Lag: {}\n".format(self.ngrc_settings["time_lag"]) +
            "solver: {}".format(self.ngrc_settings["wout_solver"])
        )


if __name__ == "__main__":
    lorenz = Lorenz63(0.025, 5, 10, 120)
    dataset_settings1 = {
        "test": 1
    }
    settings = {
        "normed": False,
        "time_lag": 2,
        "start": 1000,
        "length": 5000,
        "ratio": 0.8,
        "ridge_param": 1e-6,
        "wout_solver": "new2"
    }
    # s = NGRCSolver(SantaFe(), dataset_settings1, ngrc_settings1)
    s = NGRCSolver(Lorenz63(0.025, 5, 10, 120), settings)
    s.run()






























