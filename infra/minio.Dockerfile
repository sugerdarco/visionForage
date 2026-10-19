FROM alpine:3.23.4@sha256:5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11

# Context is artifacts/minio, populated by build_storage_tools.sh (see LOCAL.md).
WORKDIR /opt/minio
COPY minio mc LICENSE-minio LICENSE-mc SHA256SUMS ./
RUN sha256sum -c SHA256SUMS \
    && ln -s /opt/minio/minio /usr/local/bin/minio \
    && ln -s /opt/minio/mc /usr/local/bin/mc \
    && mkdir /data && chown 1000:1000 /data
USER 1000:1000
ENV HOME=/tmp
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
