FROM gcr.io/oss-fuzz-base/base-builder-rust
COPY build.sh $SRC/
RUN git clone --depth 1 https://github.com/coder-11125/rust-compress $SRC/rust-compress
WORKDIR $SRC/rust-compress
