amd64:
	sudo docker build  --platform linux/amd64   -t unipd_pod_ca23:amd64 .

arm64:
	sudo docker build  --platform linux/arm64   -t unipd_pod_ca23:arm64 .


arm64-push: arm64
	sudo docker tag unipd_pod_ca23:arm64 iogiul/unipd_pod_ca23:arm64
	sudo docker push iogiul/unipd_pod_ca23:arm64


amd64-push: amd64
	sudo docker tag unipd_pod_ca23:amd64 iogiul/unipd_pod_ca23:amd64
	sudo docker push iogiul/unipd_pod_ca23:amd64