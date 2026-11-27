export async function requestTooLarge(request: Request, limit = 262144): Promise<boolean> {
  if (Number(request.headers.get('content-length') ?? 0) > limit) return true;
  if (!request.body) return false;
  const reader = request.clone().body!.getReader();
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return false;
      size += value.byteLength;
      if (size > limit) return true;
    }
  } finally {
    void reader.cancel();
  }
}
