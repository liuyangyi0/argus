import { flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MSEPlayer, PlayerFactory, WebRTCPlayer } from '../PlayerFactory'

class FakeSocket {
  static instances: FakeSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close = vi.fn()
  send = vi.fn()
  constructor(_url: string) { FakeSocket.instances.push(this) }
}

class FakePeer {
  static instances: FakePeer[] = []
  ontrack: ((event: { streams: MediaStream[] }) => void) | null = null
  onconnectionstatechange: (() => void) | null = null
  onicecandidate: ((event: RTCPeerConnectionIceEvent) => void) | null = null
  close = vi.fn()
  addTransceiver = vi.fn()
  createOffer = vi.fn().mockResolvedValue({ type: 'offer', sdp: 'test' })
  setLocalDescription = vi.fn().mockResolvedValue(undefined)
  setRemoteDescription = vi.fn().mockResolvedValue(undefined)
  addIceCandidate = vi.fn().mockResolvedValue(undefined)
  constructor(_options: unknown) { FakePeer.instances.push(this) }
}

class FakeMediaSource extends EventTarget {
  readyState = 'closed'
}

function stream() {
  const stop = vi.fn()
  return { media: { getTracks: () => [{ stop }] } as unknown as MediaStream, stop }
}

describe('player media ownership and cancellation', () => {
  const NativeURL = globalThis.URL
  let revoke: ReturnType<typeof vi.fn<(url: string) => void>>

  beforeEach(() => {
    FakeSocket.instances = []
    FakePeer.instances = []
    revoke = vi.fn<(url: string) => void>()
    class TestURL extends NativeURL {
      static createObjectURL = vi.fn().mockReturnValue('blob:old-player')
      static revokeObjectURL = revoke
    }
    vi.stubGlobal('URL', TestURL)
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal('RTCPeerConnection', FakePeer)
    vi.stubGlobal('MediaSource', FakeMediaSource)
    vi.stubGlobal('RTCSessionDescription', class { constructor(_init: unknown) {} })
    vi.stubGlobal('RTCIceCandidate', class { constructor(_init: unknown) {} })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('stops only its own WebRTC tracks without detaching a newer owner', async () => {
    const video = document.createElement('video')
    const old = stream()
    const newer = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [old.media] })
    await expect(pending).resolves.toBe(true)
    video.srcObject = newer.media

    player.destroy()

    expect(old.stop).toHaveBeenCalledOnce()
    expect(newer.stop).not.toHaveBeenCalled()
    expect(video.srcObject).toBe(newer.media)
  })

  it('requests recovery instead of reporting playing after a connected WebRTC socket error clears media', async () => {
    const video = document.createElement('video')
    const owned = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onStatusChange = vi.fn()
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [owned.media] })
    await expect(pending).resolves.toBe(true)
    expect(player.status).toBe('playing')

    FakeSocket.instances[0]?.onerror?.()

    expect(player.status).toBe('error')
    expect(player.onStatusChange).toHaveBeenLastCalledWith('error')
    expect(player.onReconnectRequest).toHaveBeenCalledOnce()
    expect(video.srcObject).toBeNull()
    expect(owned.stop).toHaveBeenCalledOnce()
    player.destroy()
    player.destroy()
    expect(player.onReconnectRequest).toHaveBeenCalledOnce()
    expect(owned.stop).toHaveBeenCalledOnce()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles an initial WebRTC failure without requesting an established-player reconnect', async () => {
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onerror?.()
    await expect(pending).resolves.toBe(false)
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    player.destroy()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles a synchronous peer constructor failure immediately and clears its timeout', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('RTCPeerConnection', class {
      constructor() { throw new Error('WebRTC capability unavailable') }
    })
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    expect(vi.getTimerCount()).toBe(1)
    expect(() => FakeSocket.instances[0]?.onopen?.()).not.toThrow()
    await expect(pending).resolves.toBe(false)
    expect(vi.getTimerCount()).toBe(0)
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    player.destroy()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles a synchronous transceiver failure and cleans the partially created peer', async () => {
    vi.useFakeTimers()
    class UnsupportedPeer extends FakePeer {
      addTransceiver = vi.fn(() => { throw new Error('unsupported transceiver') })
    }
    vi.stubGlobal('RTCPeerConnection', UnsupportedPeer)
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    expect(() => FakeSocket.instances[0]?.onopen?.()).not.toThrow()
    await expect(pending).resolves.toBe(false)
    expect(vi.getTimerCount()).toBe(0)
    expect(FakePeer.instances[0]?.close).toHaveBeenCalledOnce()
    expect(FakePeer.instances[0]?.createOffer).not.toHaveBeenCalled()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
  })

  it.each(['answer', 'candidate'])('ignores a late rejected WebRTC %s after cancellation without disturbing the next media owner', async kind => {
    let rejectOperation!: (reason: Error) => void
    const operation = new Promise<void>((_resolve, reject) => { rejectOperation = reject })
    class SlowPeer extends FakePeer {
      setRemoteDescription = vi.fn().mockReturnValue(operation)
      addIceCandidate = vi.fn().mockReturnValue(operation)
    }
    vi.stubGlobal('RTCPeerConnection', SlowPeer)
    const video = document.createElement('video')
    const old = stream()
    const newer = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [old.media] })
    await expect(pending).resolves.toBe(true)
    FakeSocket.instances[0]?.onmessage?.({ data: JSON.stringify({ type: `webrtc/${kind}`, value: 'negotiation' }) } as MessageEvent)
    player.destroy()
    video.srcObject = newer.media
    rejectOperation(new Error('peer was closed during negotiation'))
    await flushPromises()

    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    expect(video.srcObject).toBe(newer.media)
    expect(newer.stop).not.toHaveBeenCalled()
    expect(old.stop).toHaveBeenCalledOnce()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('settles a rejected initial remote description without an unhandled negotiation failure', async () => {
    class RejectedPeer extends FakePeer {
      setRemoteDescription = vi.fn().mockRejectedValue(new Error('invalid answer'))
    }
    vi.stubGlobal('RTCPeerConnection', RejectedPeer)
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakeSocket.instances[0]?.onmessage?.({ data: JSON.stringify({ type: 'webrtc/answer', value: 'bad-answer' }) } as MessageEvent)
    await expect(pending).resolves.toBe(false)
    expect(player.onReconnectRequest).not.toHaveBeenCalled()
    expect(FakePeer.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('requests recovery for a current ICE failure after playback has started', async () => {
    class RejectedPeer extends FakePeer {
      addIceCandidate = vi.fn().mockRejectedValue(new Error('candidate rejected'))
    }
    vi.stubGlobal('RTCPeerConnection', RejectedPeer)
    const video = document.createElement('video')
    const owned = stream()
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
    player.onReconnectRequest = vi.fn()
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    FakePeer.instances[0]?.ontrack?.({ streams: [owned.media] })
    await expect(pending).resolves.toBe(true)
    FakeSocket.instances[0]?.onmessage?.({ data: JSON.stringify({ type: 'webrtc/candidate', value: 'bad-candidate' }) } as MessageEvent)
    await flushPromises()
    expect(player.status).toBe('error')
    expect(player.onReconnectRequest).toHaveBeenCalledOnce()
    expect(owned.stop).toHaveBeenCalledOnce()
  })

  it('fails malformed WebRTC messages through controlled cleanup', async () => {
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    expect(() => FakeSocket.instances[0]?.onmessage?.({ data: '{broken' } as MessageEvent)).not.toThrow()
    await expect(pending).resolves.toBe(false)
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('releases only its MSE URL without resetting the next player element', async () => {
    const video = document.createElement('video')
    const load = vi.spyOn(video, 'load').mockImplementation(() => {})
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    expect(video.getAttribute('src')).toBe('blob:old-player')
    video.src = 'blob:new-player'
    player.destroy()

    await expect(pending).resolves.toBe(false)
    expect(video.getAttribute('src')).toBe('blob:new-player')
    expect(load).not.toHaveBeenCalled()
    expect(revoke).toHaveBeenCalledOnce()
    expect(revoke).toHaveBeenCalledWith('blob:old-player')
  })

  it('does not clear a newer WebRTC stream even if its old MSE src attribute remains', async () => {
    const video = document.createElement('video')
    const load = vi.spyOn(video, 'load').mockImplementation(() => {})
    const newer = stream()
    const player = new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    video.srcObject = newer.media
    player.destroy()

    await expect(pending).resolves.toBe(false)
    expect(video.srcObject).toBe(newer.media)
    expect(newer.stop).not.toHaveBeenCalled()
    expect(load).not.toHaveBeenCalled()
    expect(revoke).toHaveBeenCalledWith('blob:old-player')
  })

  it.each(['webrtc', 'mse'])('settles and clears the pending %s timeout on destroy', async (protocol) => {
    vi.useFakeTimers()
    const video = document.createElement('video')
    vi.spyOn(video, 'load').mockImplementation(() => {})
    const player = protocol === 'webrtc'
      ? new WebRTCPlayer('cam1', 'ws://example.test/webrtc', video)
      : new MSEPlayer('cam1', 'ws://example.test/mse', video)
    const pending = player.connect()
    expect(vi.getTimerCount()).toBe(1)
    player.destroy()
    await expect(pending).resolves.toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('aborts a pending factory connection instead of waiting for its network timeout', async () => {
    vi.spyOn(PlayerFactory, 'fetchStreamInfo').mockResolvedValue({
      camera_id: 'cam1', go2rtc: true, webrtc_ws: 'ws://example.test/webrtc', fallback: '/fallback',
    })
    const controller = new AbortController()
    const pending = PlayerFactory.create('cam1', document.createElement('video'), () => false, undefined, controller.signal)
    await flushPromises()
    expect(FakeSocket.instances).toHaveLength(1)
    controller.abort()

    await expect(pending).resolves.toBeNull()
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalledOnce()
  })

  it('does not negotiate on a peer whose offer resolves after cancellation', async () => {
    let resolveOffer!: (offer: RTCSessionDescriptionInit) => void
    const offer = new Promise<RTCSessionDescriptionInit>(resolve => { resolveOffer = resolve })
    class SlowPeer extends FakePeer {
      createOffer = vi.fn().mockReturnValue(offer)
    }
    vi.stubGlobal('RTCPeerConnection', SlowPeer)
    const player = new WebRTCPlayer('cam1', 'ws://example.test/webrtc', document.createElement('video'))
    const pending = player.connect()
    FakeSocket.instances[0]?.onopen?.()
    const peer = FakePeer.instances[0]
    player.destroy()
    await expect(pending).resolves.toBe(false)
    resolveOffer({ type: 'offer', sdp: 'late-offer' })
    await flushPromises()

    expect(peer?.setLocalDescription).not.toHaveBeenCalled()
    expect(FakeSocket.instances[0]?.send).not.toHaveBeenCalled()
  })
})
