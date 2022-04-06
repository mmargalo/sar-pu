for outer_epoch in range(n_outer_epochs):
    # Train the classification and propensity models
    for inner_epoch in range(n_inner_epochs):
        for n_iter, (x, s) in enumerate(inner_data_loader, 1):
            features = backbone(x).detach()  # DO NOT OMIT .detach()

            expected_prior_y1 = classification_model(features)
            expected_propensity = propensity_model(features)
            expected_posterior_y1 = expectation_y(expected_prior_y1, expected_propensity, s)

            optimizer_prp.zero_grad()
            optimizer_cls.zero_grad()

            loss = criterion_prp(expected_propensity, s, sample_weight=expected_posterior_y1)
            loss.backward()
            optimizier_prp.step()

            classification_s = ...
            classification_weight = ...
            expected_prior_y1 = torch.cat?
            loss = criterion_cls(expected_prior_y1, classification_s, sample_weight=classification_weight)
            loss.backward()
            optimizier_cls.step()

    # Train the backbone/encoder model
    for param in classification_model.parameters():
        param.requires_grad_(False)

    for param in propensity_model.parameters():
        param.requires_grad_(False)

    for n_iter, x in enumerate(train_loader, 1):
        features = backbone(x)

        expected_prior_y1 = classification_model(features)
        expected_propensity = propensity_model(features)

        loss = -loglikelihood_probs(expected_prior_y1, expected_propensity, s)
        loss.backward()
        optimizier.step()
