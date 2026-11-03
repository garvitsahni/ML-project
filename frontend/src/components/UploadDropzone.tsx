import { useCallback, useRef, useState } from 'react'
import { Upload } from 'lucide-react'

interface UploadDropzoneProps {
  file: File | null
  onFileSelect: (file: File) => void
  acceptedTypes?: string
  maxSize?: number
  hint?: string
}

export function UploadDropzone({
  file,
  onFileSelect,
  acceptedTypes = 'image/*',
  maxSize = 10 * 1024 * 1024,
  hint,
}: UploadDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const accept = useCallback(
    (candidate: File) => {
      setLocalError(null)
      if (candidate.size > maxSize) {
        setLocalError(
          `File is ${(candidate.size / 1024 / 1024).toFixed(1)} MB; limit is ` +
            `${(maxSize / 1024 / 1024).toFixed(0)} MB.`,
        )
        return
      }
      onFileSelect(candidate)
    },
    [maxSize, onFileSelect],
  )

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      setIsDragOver(false)
      const dropped = event.dataTransfer.files?.[0]
      if (dropped) accept(dropped)
    },
    [accept],
  )

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click()
        }}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        className={[
          'flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 text-center transition-colors',
          isDragOver
            ? 'border-accent bg-[#fdf1ee]'
            : 'border-line bg-paper/60 hover:border-faint hover:bg-paper',
        ].join(' ')}
      >
        <Upload size={30} className="mb-3 text-faint" aria-hidden />
        {file ? (
          <>
            <p className="font-medium text-ink">{file.name}</p>
            <p className="mt-0.5 text-sm text-muted">
              {(file.size / 1024 / 1024).toFixed(2)} MB — click to replace
            </p>
          </>
        ) : (
          <>
            <p className="font-medium text-ink">Drop a file here or click to browse</p>
            <p className="mt-1 text-sm text-muted">{hint ?? 'Images only'}</p>
            <p className="mt-1 text-xs text-faint">
              Max {(maxSize / 1024 / 1024).toFixed(0)} MB
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={acceptedTypes}
          className="hidden"
          onChange={(event) => {
            const selected = event.target.files?.[0]
            if (selected) accept(selected)
            event.target.value = ''
          }}
        />
      </div>
      {localError && (
        <p className="mt-2 text-sm text-crowded" role="alert">
          {localError}
        </p>
      )}
    </div>
  )
}
