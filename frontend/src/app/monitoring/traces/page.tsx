"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ArrowLeft } from "lucide-react";
import { useTraces, useTraceDetail } from "@/lib/queries";
import type { PaginationParams } from "@/types";

function TraceDetailDialog({
  traceId,
  open,
  onOpenChange,
}: {
  traceId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { data: trace } = useTraceDetail(traceId);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="!left-4 !right-4 !top-4 !w-auto !max-w-none !translate-x-0 !translate-y-0 flex max-h-[calc(100dvh-2rem)] flex-col overflow-hidden p-0 sm:!left-1/2 sm:!right-auto sm:!top-1/2 sm:!w-full sm:!max-w-2xl sm:!translate-x-[-50%] sm:!translate-y-[-50%] sm:max-h-[85vh]">
        <DialogHeader className="shrink-0 px-6 pt-6">
          <DialogTitle>트레이스 상세</DialogTitle>
        </DialogHeader>
        {trace ? (
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-6 pb-6">
            <div className="space-y-2 text-sm">
              <div className="flex items-start justify-between gap-3">
                <span className="text-muted-foreground">쿼리</span>
                <span className="max-w-[70%] break-all text-right font-medium">
                  {trace.query || "-"}
                </span>
              </div>
              <div className="flex items-start justify-between gap-3">
                <span className="text-muted-foreground">출력</span>
                <span className="max-w-[70%] break-all text-right font-medium">
                  {trace.output || "-"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">총 소요시간</span>
                <span className="font-medium">{(trace.total_duration_ms / 1000).toFixed(2)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">상태</span>
                <Badge variant={trace.status === "success" ? "default" : "destructive"}>
                  {trace.status === "success" ? "성공" : "오류"}
                </Badge>
              </div>
            </div>

            {/* Span timeline */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Span 타임라인</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {trace.spans?.length ? trace.spans.map((span, i) => {
                    const widthPercent =
                      trace.total_duration_ms > 0
                        ? Math.min(100, Math.max(5, (span.duration_ms / trace.total_duration_ms) * 100))
                        : 100;
                    return (
                      <div key={i} className="space-y-1">
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span className="truncate font-medium">{span.name}</span>
                          <span className="shrink-0 text-muted-foreground">{span.duration_ms}ms</span>
                        </div>
                        <div className="h-2 rounded-full bg-muted">
                          <div
                            className="h-2 rounded-full bg-primary"
                            style={{ width: `${widthPercent}%` }}
                          />
                        </div>
                      </div>
                    );
                  }) : (
                    <p className="text-xs text-muted-foreground">스팬 정보가 없습니다.</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <p className="px-6 pb-6 text-muted-foreground">로딩 중...</p>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default function TracesPage() {
  const [params, setParams] = useState<PaginationParams>({ page: 1, size: 20 });
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const { data, isLoading } = useTraces(params);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" asChild>
          <Link href="/monitoring">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <h2 className="text-2xl font-bold">트레이스</h2>
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>시간</TableHead>
              <TableHead>쿼리</TableHead>
              <TableHead>출력</TableHead>
              <TableHead className="text-right">소요시간</TableHead>
              <TableHead className="text-right">상태</TableHead>
              <TableHead className="text-right">작업</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={6} className="h-24 text-center">
                  로딩 중...
                </TableCell>
              </TableRow>
            ) : !data?.items?.length ? (
              <TableRow>
                <TableCell colSpan={6} className="h-24 text-center">
                  트레이스가 없습니다.
                </TableCell>
              </TableRow>
            ) : (
              data.items.map((trace) => (
                <TableRow key={trace.id}>
                  <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                    {new Date(trace.created_at).toLocaleString("ko-KR")}
                  </TableCell>
                  <TableCell className="max-w-[300px] truncate text-sm">
                    {trace.query}
                  </TableCell>
                  <TableCell className="max-w-[360px] truncate text-sm text-muted-foreground">
                    {trace.output || "-"}
                  </TableCell>
                  <TableCell className="text-right text-sm">
                    {(trace.total_duration_ms / 1000).toFixed(2)}s
                  </TableCell>
                  <TableCell className="text-right">
                    <Badge variant={trace.status === "success" ? "default" : "destructive"}>
                      {trace.status === "success" ? "성공" : "오류"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedTraceId(trace.id)}
                    >
                      상세
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Total count */}
      {data && data.total > 0 && (
        <p className="text-sm text-muted-foreground">
          총 {data.total}개
        </p>
      )}

      {selectedTraceId && (
        <TraceDetailDialog
          traceId={selectedTraceId}
          open={!!selectedTraceId}
          onOpenChange={(open) => !open && setSelectedTraceId(null)}
        />
      )}
    </div>
  );
}
