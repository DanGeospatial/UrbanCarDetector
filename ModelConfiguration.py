class MyFasterRCNN:
    """
    Create class with following fixed function names and the number of arguents to train your model from external source
    """

    try:
        import torch
        import torchvision
        import fastai

        tvisver = [int(x) for x in torchvision.__version__.split(".")]
    except:
        pass

    def get_model(self, data, backbone=None, **kwargs):
        """
        In this fuction you have to define your model with following two arguments!

        data - Object returned from prepare_data method(Fastai databunch)

        These two arguments comes from dataset which you have prepared from prepare_data method above.

        """
        (
            self.fasterrcnn_kwargs,
            kwargs,
        ) = self.fastai.core.split_kwargs_by_func(
            kwargs, self.torchvision.models.detection.FasterRCNN.__init__
        )

        if "timm" in backbone:
            from arcgis.learn.models._timm_utils import timm_config, _get_feature_size

            backbone_cut = timm_config(backbone)["cut"]
        else:
            backbone_cut = None

        if backbone is None:
            backbone = self.torchvision.models.resnet50

        elif type(backbone) is str:
            if hasattr(self.torchvision.models, backbone):
                backbone = getattr(self.torchvision.models, backbone)
            elif hasattr(self.torchvision.models.detection, backbone):
                backbone = getattr(self.torchvision.models.detection, backbone)
            elif "timm:" in backbone:
                import timm

                bckbn = backbone.split(":")[1]
                if hasattr(timm.models, bckbn):
                    backbone = getattr(timm.models, bckbn)
        else:
            backbone = backbone
        pretrained_backbone = kwargs.get("pretrained_backbone", True)
        assert type(pretrained_backbone) == bool
        if backbone.__name__ == "resnet50" and "timm" not in backbone.__module__:
            model = self.torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=pretrained_backbone,
                pretrained_backbone=False,
                min_size=1.5 * data.chip_size,
                max_size=2 * data.chip_size,
                **self.fasterrcnn_kwargs,
            )

        elif (
            backbone.__name__ in ["resnet101", "resnet152"]
            and "timm" not in backbone.__module__
        ):
            backbone_fpn = (
                self.torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                    backbone.__name__, pretrained=pretrained_backbone
                )
            )
            model = self.torchvision.models.detection.FasterRCNN(
                backbone_fpn,
                91,
                min_size=1.5 * data.chip_size,
                max_size=2 * data.chip_size,
                **self.fasterrcnn_kwargs,
            )

        else:
            backbone_small = self.fastai.vision.learner.create_body(
                backbone, pretrained_backbone, backbone_cut
            )
            if "timm" in backbone.__module__:
                from arcgis.learn.models._maskrcnn import TimmFPNBackbone

                try:
                    backbone_small = TimmFPNBackbone(backbone_small, data.chip_size)
                except:
                    pass

            if not hasattr(backbone_small, "out_channels"):
                if "tresnet" in backbone.__module__:
                    backbone_small.out_channels = _get_feature_size(
                        backbone, backbone_cut
                    )[-1][1]
                else:
                    backbone_small.out_channels = (
                        self.fastai.callbacks.hooks.num_features_model(
                            self.torch.nn.Sequential(*backbone_small.children())
                        )
                    )

            model = self.torchvision.models.detection.FasterRCNN(
                backbone_small,
                91,
                min_size=1.5 * data.chip_size,
                max_size=2 * data.chip_size,
                **self.fasterrcnn_kwargs,
            )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            self.torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, len(data.classes)
            )
        )

        if data._is_multispectral:
            model.transform.image_mean = [0] * len(data._extract_bands)
            model.transform.image_std = [1] * len(data._extract_bands)

        model.roi_heads.nms_thresh = 0.1
        model.roi_heads.score_thresh = 0.2

        self.model = model

        return model

    def on_batch_begin(self, learn, model_input_batch, model_target_batch, **kwargs):
        """
        This fuction is dedicated to put the inputs and outputs of the model before training. This is equivalent to fastai
        on_batch_begin function. In this function you will get the inputs and targets with applied transormations. You should
        be very carefull to return the model input and target during traing, model will only accept model_input(in many cases it
        is possible to model accept input and target both to return the loss during traing and you don't require to compute loss
        from the model output and the target by yourself), if you want to compute the loss by yourself by taking the output of the
        model and targets then you have to return the model_target in desired format to calculate loss in the loss function.

        learn - Fastai learner object.
        model_input_batch - transformed input batch(images) with tensor shape [N,C,H,W].
        model_target_batch - transformed target batch. list with [bboxes, classes]. Where bboxes tensor shape will be
                            [N, maximum_num_of_boxes_pesent_in_one_image_of_the_batch, 4(y1,x1,y2,x2 fastai default bbox
                            formate)] and bboxes in the range from -1 to 1(default fastai formate), and classes is the tenosr
                            of shape [N, maximum_num_of_boxes_pesent_in_one_image_of_the_batch] which represents class of each
                            bboxes.
        if you are synthesizing new data from the model_target_batch and model_input_batch, in that case you need to put
        your data on correct device.

        return model_input and model_target from this function.

        """

        # during training after each epoch, validation loss is required on validation set of datset.
        # torchvision FasterRCNN model gives losses only on training mode that is why set your model in train mode
        # such that you can get losses for your validation datset as well after each epoch.
        train = kwargs.get("train")
        learn.model.train()
        if train:
            self.model.roi_heads.train_val = False
            self.model.rpn.train_val = False
            self.model.train_val = False
            self.model.transform.train_val = False
        else:
            learn.model.backbone.eval()  # to get feature in eval mode for evaluation
            self.model.roi_heads.train_val = True
            self.model.rpn.train_val = True
            self.model.train_val = True
            self.model.transform.train_val = True

        target_list = []

        # denormalize from imagenet_stats
        if not learn.data._is_multispectral:
            imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
            mean = self.torch.tensor(imagenet_stats[0], dtype=self.torch.float32).to(
                model_input_batch.device
            )
            std = self.torch.tensor(imagenet_stats[1], dtype=self.torch.float32).to(
                model_input_batch.device
            )
            model_input_batch = (
                model_input_batch.permute(0, 2, 3, 1) * std + mean
            ).permute(0, 3, 1, 2)

        for bbox, label in zip(*model_target_batch):
            bbox = (
                (bbox + 1) / 2
            ) * learn.data.chip_size  # FasterRCNN model require bboxes with values between 0 and H and 0 and W.
            mask = (bbox[:, 2:] >= (bbox[:, :2] + 1.0)).all(1)
            bbox = bbox[mask]
            label = label[mask]
            target = (
                {}
            )  # FasterRCNN require target of each image in the formate of dictionary.
            # If image comes without any bboxes.
            if (self.tvisver[0] == 0 and self.tvisver[1] < 6) and bbox.nelement() == 0:
                bbox = self.torch.tensor([[0.0, 0.0, 0.0, 0.0]]).to(learn.data.device)
                label = self.torch.tensor([0]).to(learn.data.device)
            # FasterRCNN require the formate of bboxes [x1,y1,x2,y2].
            bbox = self.torch.index_select(
                bbox,
                1,
                self.torch.tensor([1, 0, 3, 2]).to(learn.data.device),
            )
            target["boxes"] = bbox
            target["labels"] = label
            target_list.append(
                target
            )  # FasterRCNN require batches target in form of list of dictionary.

        # handle batch size one in training
        if model_input_batch.shape[0] < 2:
            model_input_batch = self.torch.cat((model_input_batch, model_input_batch))
            target_list.append(target_list[0])

        # FasterRCNN require model input with images and coresponding targets in training mode to return the losses so append
        # the targets in model input itself.
        model_input = [list(model_input_batch), target_list]
        # Model target is not required in traing mode so just return the same model_target to train the model.
        model_target = model_target_batch

        # return model_input and model_target
        return model_input, model_target

    def transform_input(self, xb, thresh=0.5, nms_overlap=0.1):  # transform_input
        """
        function for feding the input to the model in validation/infrencing mode.

        xb - tensor with shape [N, C, H, W]
        """
        self.model.roi_heads.train_val = False
        self.model.rpn.train_val = False
        self.model.train_val = False
        self.model.transform.train_val = False
        self.nms_thres = self.model.roi_heads.nms_thresh
        self.thresh = self.model.roi_heads.score_thresh
        self.model.roi_heads.nms_thresh = nms_overlap
        self.model.roi_heads.score_thresh = thresh

        # denormalize from imagenet_stats
        imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        mean = self.torch.tensor(imagenet_stats[0], dtype=self.torch.float32).to(
            xb.device
        )
        std = self.torch.tensor(imagenet_stats[1], dtype=self.torch.float32).to(
            xb.device
        )

        xb = (xb.permute(0, 2, 3, 1) * std + mean).permute(0, 3, 1, 2)

        return list(xb)  # model input require in the formate of list

    def transform_input_multispectral(self, xb, thresh=0.5, nms_overlap=0.1):
        self.model.roi_heads.train_val = False
        self.model.rpn.train_val = False
        self.model.train_val = False
        self.model.transform.train_val = False
        self.nms_thres = self.model.roi_heads.nms_thresh
        self.thresh = self.model.roi_heads.score_thresh
        self.model.roi_heads.nms_thresh = nms_overlap
        self.model.roi_heads.score_thresh = thresh

        return list(xb)

    def loss(self, model_output, *model_target):
        """
        Define loss in this function.

        model_output - model output after feding input to the model in traing mode.
        *model_target - targets of the model which you have return in above on_batch_begin function.

        return loss for the model
        """
        if isinstance(model_output, tuple):
            model_output = model_output[1]
        # FasterRCNN model return loss in traing mode by feding input to the model it does not require target to compute the loss
        final_loss = 0.0
        for i in model_output.values():
            i[self.torch.isnan(i)] = 0.0
            i[self.torch.isinf(i)] = 0.0
            final_loss += i

        return final_loss

    def post_process(self, pred, nms_overlap, thres, chip_size, device):
        """
        Fuction dedicated for post processing your output of the model in validation/infrencing mode.

        pred - Predictions(output) of the model after feding the batch of input image.
        nms_overlap - If your model post processing require nms_overlap.
        thres - detction thresold if required in post processing.
        chip_size - If chip_size required in model post processing.
        device - device on which you should put you output after post processing.

        It should return the bboxes in range -1 to 1 and the formate of the post processed result is list of tuple for each
        image and tuple should contain (bboxes, label, score) for each image. bboxes should be the tensor of shape
        [Number_of_bboxes_in_image, 4], label should be the tensor of shape[Number_of_bboxes_in_image,] and score should be
        the tensor of shape[Number_of_bboxes_in_image,].
        """
        if not self.model.roi_heads.train_val:
            self.model.roi_heads.score_thresh = self.thresh
            self.model.roi_heads.nms_thresh = self.nms_thres

        post_processed_pred = []
        for p in pred:
            bbox, label, score = p["boxes"], p["labels"], p["scores"]
            # convert bboxes in range -1 to 1.
            bbox = bbox / (chip_size / 2) - 1
            # convert bboxes in format [y1,x1,y2,x2]
            bbox = self.torch.index_select(
                bbox, 1, self.torch.tensor([1, 0, 3, 2]).to(bbox.device)
            )
            # Append the tuple in list for each image
            post_processed_pred.append(
                (bbox.data.to(device), label.to(device), score.to(device))
            )

        return post_processed_pred
