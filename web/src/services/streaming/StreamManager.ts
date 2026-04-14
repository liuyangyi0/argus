export class StreamManager {
  static MAX_STREAMS = 4;
  static activeCount = 0;
  
  static requestSlot(): boolean {
    if (this.activeCount >= this.MAX_STREAMS) return false;
    this.activeCount++;
    return true;
  }
  
  static releaseSlot() {
    this.activeCount = Math.max(0, this.activeCount - 1);
  }
}
