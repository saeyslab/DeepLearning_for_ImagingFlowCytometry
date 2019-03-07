
    def predict():
        prerun(exp=False)

        ds, ds_len = preprocessing.load_dataset(None, None, meta, args)
        
        m = keras.models.load_model(args["model_hdf5"], compile=False)
        m.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[bal_acc]
        )
        
        m.evaluate(
            ds,
            batch_size=args["batch_size"],
            steps_per_epoch=int(np.ceil(ds_len/args["batch_size"])),
        )