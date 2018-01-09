#! /users/guruprakash.r/miniconda2/bin/python

def validationLoss(sequences):
	h = net.init_hidden()
	final_loss = 0
	for k, seq in enumerate(sequences):
		chunk = seq
		xs, zs = chunk[:-1], chunk[1:]
		loss = 0

		net.zero_grad()
		for x, z in zip(xs, zs):

			x = np.array(x, dtype=float)
			x = torch.from_numpy(x)
			x = x.float()

			z = np.array(z, dtype=float)
			z = torch.from_numpy(z)
			z = z.float()

			x = Variable(x, requires_grad=False)
			z = Variable(z)

			y, h = net(x, h)
			loss += criterion(y.view(1, -1), z)

		# Saving h again, so it's not consumed by .backward() ahead.
		final_loss += loss
		h = h.data
		h = Variable(h, requires_grad=True)
	return final_loss