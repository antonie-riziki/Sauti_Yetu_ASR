class PCMProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        // Mono channel
        const pcm = new Float32Array(input[0]);
        this.port.postMessage(pcm.buffer);

        return true;
    }
}

registerProcessor("pcm-processor", PCMProcessor);
