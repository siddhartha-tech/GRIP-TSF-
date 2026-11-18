"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge, StatusBadge } from "../../components/shared";
import { PROMOTIONS, CAMPAIGNS, formatDate } from "../../data/store";
import { ChevronLeft, ChevronRight, Filter } from "lucide-react";

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

const PROMO_COLORS = [
  "bg-emerald-200 text-emerald-800 border-emerald-300",
  "bg-blue-200 text-blue-800 border-blue-300",
  "bg-purple-200 text-purple-800 border-purple-300",
  "bg-amber-200 text-amber-800 border-amber-300",
  "bg-red-200 text-red-800 border-red-300",
  "bg-pink-200 text-pink-800 border-pink-300",
  "bg-teal-200 text-teal-800 border-teal-300",
  "bg-indigo-200 text-indigo-800 border-indigo-300",
];

export default function CalendarPage() {
  const [currentMonth, setCurrentMonth] = useState(10); // November (0-indexed)
  const [currentYear, setCurrentYear] = useState(2026);
  const [siteFilter, setSiteFilter] = useState<string>("all");

  const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
  const firstDayOfWeek = new Date(currentYear, currentMonth, 1).getDay();

  const filteredPromos = PROMOTIONS.filter((p) => {
    if (siteFilter !== "all" && !p.sites.includes(siteFilter as any)) return false;
    const start = new Date(p.startsAt);
    const end = new Date(p.endsAt);
    const monthStart = new Date(currentYear, currentMonth, 1);
    const monthEnd = new Date(currentYear, currentMonth + 1, 0);
    return start <= monthEnd && end >= monthStart;
  });

  const getPromosForDay = (day: number) => {
    const date = new Date(currentYear, currentMonth, day);
    return filteredPromos.filter((p) => {
      const start = new Date(p.startsAt);
      const end = new Date(p.endsAt);
      return date >= new Date(start.getFullYear(), start.getMonth(), start.getDate()) &&
             date <= new Date(end.getFullYear(), end.getMonth(), end.getDate());
    });
  };

  const prevMonth = () => {
    if (currentMonth === 0) {
      setCurrentMonth(11);
      setCurrentYear(currentYear - 1);
    } else {
      setCurrentMonth(currentMonth - 1);
    }
  };

  const nextMonth = () => {
    if (currentMonth === 11) {
      setCurrentMonth(0);
      setCurrentYear(currentYear + 1);
    } else {
      setCurrentMonth(currentMonth + 1);
    }
  };

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Promotion Calendar"
          subtitle="Scheduled promotions by site"
        />

        {/* Controls */}
        <div className="card p-4 mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button onClick={prevMonth} className="p-2 rounded-lg hover:bg-gray-100">
              <ChevronLeft size={18} />
            </button>
            <h2 className="text-lg font-bold text-gray-900 w-48 text-center">
              {MONTHS[currentMonth]} {currentYear}
            </h2>
            <button onClick={nextMonth} className="p-2 rounded-lg hover:bg-gray-100">
              <ChevronRight size={18} />
            </button>
          </div>
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-gray-400" />
            <select
              value={siteFilter}
              onChange={(e) => setSiteFilter(e.target.value)}
              className="input-field w-44 text-xs"
            >
              <option value="all">All Sites</option>
              <option value="basspro.com">basspro.com</option>
              <option value="cabelas.com">cabelas.com</option>
              <option value="basspro.ca">basspro.ca</option>
              <option value="mackspw.com">mackspw.com</option>
            </select>
          </div>
        </div>

        {/* Calendar Grid */}
        <div className="card overflow-hidden">
          {/* Day headers */}
          <div className="grid grid-cols-7 bg-gray-50 border-b border-gray-200">
            {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
              <div key={day} className="px-2 py-3 text-center text-xs font-semibold text-gray-500">
                {day}
              </div>
            ))}
          </div>

          {/* Calendar cells */}
          <div className="grid grid-cols-7">
            {/* Empty cells for days before the 1st */}
            {Array.from({ length: firstDayOfWeek }).map((_, i) => (
              <div key={`empty-${i}`} className="min-h-[100px] border-b border-r border-gray-100 bg-gray-50/50" />
            ))}

            {/* Day cells */}
            {Array.from({ length: daysInMonth }).map((_, i) => {
              const day = i + 1;
              const dayPromos = getPromosForDay(day);
              const isToday = day === 16 && currentMonth === 10 && currentYear === 2026;

              return (
                <div
                  key={day}
                  className={`min-h-[100px] border-b border-r border-gray-100 p-1.5 ${
                    isToday ? "bg-bass-cream/30" : ""
                  }`}
                >
                  <div className={`text-xs font-semibold mb-1 ${
                    isToday
                      ? "w-6 h-6 rounded-full bg-bass-gold text-white flex items-center justify-center"
                      : "text-gray-500 pl-1"
                  }`}>
                    {day}
                  </div>
                  <div className="space-y-0.5">
                    {dayPromos.slice(0, 3).map((promo, idx) => (
                      <div
                        key={promo.id}
                        className={`text-[10px] font-medium px-1.5 py-0.5 rounded border truncate ${
                          PROMO_COLORS[PROMOTIONS.indexOf(promo) % PROMO_COLORS.length]
                        }`}
                        title={promo.name}
                      >
                        {promo.name}
                      </div>
                    ))}
                    {dayPromos.length > 3 && (
                      <div className="text-[10px] text-gray-400 pl-1">
                        +{dayPromos.length - 3} more
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="card p-4 mt-4">
          <h3 className="text-xs font-semibold text-gray-500 mb-2">Active Promotions This Month</h3>
          <div className="flex flex-wrap gap-2">
            {filteredPromos.map((promo, idx) => (
              <div
                key={promo.id}
                className={`text-xs font-medium px-2 py-1 rounded border ${
                  PROMO_COLORS[PROMOTIONS.indexOf(promo) % PROMO_COLORS.length]
                }`}
              >
                {promo.name}
                <span className="ml-1.5 opacity-60">
                  {formatDate(promo.startsAt)} — {formatDate(promo.endsAt)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
